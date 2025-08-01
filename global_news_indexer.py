from dotenv import load_dotenv
import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from utils import month_map

load_dotenv()

PDF_PATH = "resources/file.pdf"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_NAME = "world-news-rag"

CLOUD = os.getenv("CLOUD", "aws")
REGION = os.getenv("REGION", "us-east-1")

def load_world_news_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    full_text = "\n".join([page.page_content for page in pages])

    pattern = re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December) 20\d{2}")
    matches = list(pattern.finditer(full_text))

    docs = []
    for idx, match in enumerate(matches):
        month_str = match.group()
        start_pos = match.end()
        end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)

        month_text = full_text[start_pos:end_pos].strip()
        content = f"{month_str}\n{month_text}"

        parts = month_str.split()
        month_name = parts[0].lower()
        year = parts[1]
        yyyymm = None
        if month_name in month_map:
            for val in month_map[month_name]:
                if val.startswith(year):
                    yyyymm = val
                    break
        if not yyyymm:
            print(f"Warning: Could not map {month_str}")
            yyyymm = month_str

        doc = Document(
            page_content=content,
            metadata={"month": yyyymm, "topic": "World News"}
        )
        docs.append(doc)

    print(f"\nExtracted {len(docs)} documents from PDF.")
    return docs

def index_documents_to_pinecone(docs):
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(cloud=CLOUD, region=REGION)
        )
        print(f"Created new index: {INDEX_NAME}")
    else:
        print(f"Using existing index: {INDEX_NAME}")

    embedder = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embedder)

    print(f"Storing {len(docs)} documents into Pinecone...")
    vectorstore.add_documents(docs)
    print("Finished uploading documents.")

if __name__ == "__main__":
    docs = load_world_news_documents(PDF_PATH)

    if docs:
        print("\n--- Sample Document ---")
        print("Metadata:", docs[0].metadata)
        print("Content (truncated):", docs[0].page_content[:300], "...")

    index_documents_to_pinecone(docs)

from dotenv import load_dotenv
import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PDF_PATH = "resources/file.pdf"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_NAME = "commodity-strategy-rag"

CLOUD = os.getenv("CLOUD", "aws")
REGION = os.getenv("REGION", "us-east-1")

def load_commodity_strategy_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    full_text = "\n".join([page.page_content for page in pages])

    gold_pattern = re.compile(r"Gold Trading Strategies", re.IGNORECASE)
    oil_pattern = re.compile(r"Crude Oil Trading Strategies", re.IGNORECASE)

    gold_match = gold_pattern.search(full_text)
    oil_match = oil_pattern.search(full_text)

    docs = []

    if gold_match and oil_match:
        gold_start = gold_match.end()
        gold_end = oil_match.start()
        oil_start = oil_match.end()
        oil_end = len(full_text)

        gold_text = full_text[gold_start:gold_end].strip()
        oil_text = full_text[oil_start:oil_end].strip()

        gold_doc = Document(
            page_content=f"Gold Trading Strategies\n{gold_text}",
            metadata={"commodity": "Gold", "topic": "Trading Strategy"}
        )
        docs.append(gold_doc)

        oil_doc = Document(
            page_content=f"Crude Oil Trading Strategies\n{oil_text}",
            metadata={"commodity": "Crude Oil", "topic": "Trading Strategy"}
        )
        docs.append(oil_doc)

    print(f"\nExtracted {len(docs)} strategy documents from PDF.")
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
    docs = load_commodity_strategy_documents(PDF_PATH)

    if docs:
        print("\n--- Sample Document ---")
        print("Metadata:", docs[0].metadata)
        print("Content (truncated):", docs[0].page_content[:300], "...")

    index_documents_to_pinecone(docs)

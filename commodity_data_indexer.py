from dotenv import load_dotenv
import os
import yfinance as yf

from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFacePipeline
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
GOOGLE_API = os.getenv("GOOGLE_API_KEY")

CLOUD = os.getenv("CLOUD", "aws")
REGION = os.getenv("REGION", "us-east-1")
INDEX_NAME = os.getenv("INDEX_NAME", "commodities-rag")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

COMMODITIES = {
    "Gold": "GC=F",
    "Crude Oil": "CL=F"
}

pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud=CLOUD, region=REGION)
    )
print(f"Pinecone index ready: {INDEX_NAME}")

embedder = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embedder)

llm_pipeline = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation'
)
model = ChatHuggingFace(llm=llm_pipeline)
# model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', api_key=GOOGLE_API)

summary_template = PromptTemplate(
    template="""You are a financial analyst. Summarize the following daily price data for {commodity} in {month}.
Write a short paragraph describing the trend, highs, lows, and any notable observations.

Data:
{data}

Summary:""",
    input_variables=["commodity", "month", "data"]
)

def generate_llm_summary(daily_text, commodity_name, month):
    prompt = summary_template.format(
        commodity=commodity_name,
        month=month,
        data=daily_text
    )
    response = model.invoke(prompt)
    return response.content.strip()

def prepare_and_store():
    all_docs = []
    for commodity, ticker in COMMODITIES.items():
        print(f"Downloading data for {commodity}")
        data = yf.Ticker(ticker).history(period="1y")

        if data.empty:
            print(f"No data for {commodity}")
            continue

        data.index = data.index.tz_localize(None)
        data['Month'] = data.index.to_period('M')

        for month, group in data.groupby('Month'):
            daily_lines = []
            for date, row in group.iterrows():
                daily_lines.append(
                    f"{date.date()}: Open={row['Open']:.2f}, High={row['High']:.2f}, "
                    f"Low={row['Low']:.2f}, Close={row['Close']:.2f}"
                )
            daily_text = "\n".join(daily_lines)
            summary = generate_llm_summary(daily_text, commodity, month)
            text = summary.split("Summary:")[-1].strip()

            doc = Document(
                page_content=text,
                metadata={"month": str(month), "commodity": commodity}
            )
            all_docs.append(doc)

    print(f"Storing {len(all_docs)} documents in Pinecone...")
    vectorstore.add_documents(all_docs)
    print("Finished indexing.")

if __name__ == "__main__":
    prepare_and_store()

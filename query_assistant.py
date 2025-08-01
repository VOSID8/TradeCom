import re

from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFacePipeline
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate

from pinecone import Pinecone
from utils import month_map, COMMODITY_MAP

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

embedder = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
pc = Pinecone(api_key=PINECONE_API_KEY)

# vectorstores
commodity_vectorstore = PineconeVectorStore(index_name="commodities-rag", embedding=embedder)
news_vectorstore = PineconeVectorStore(index_name="world-news-rag", embedding=embedder)
strategy_vectorstore = PineconeVectorStore(index_name="commodity-strategy-rag", embedding=embedder)

llm_pipeline = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation'
)
model = ChatHuggingFace(llm=llm_pipeline)

query_template = PromptTemplate(
    template="""You are a commodity trading assistant. 
Based on the following context, which includes monthly summaries, world news, and trading strategies, advise the user. 
Consider all context carefully and justify your answer.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

def extract_entities_from_query(query: str):
    commodities = []
    months = []

    query_lc = query.lower()

    for c in ["gold", "oil"]:
        if c in query_lc:
            commodities.append(COMMODITY_MAP[c])

    for word, vals in month_map.items():
        if word in query_lc:
            if "2024" in query_lc:
                months.extend([v for v in vals if v.startswith("2024")])
            elif "2025" in query_lc:
                months.extend([v for v in vals if v.startswith("2025")])
            else:
                months.extend(vals)

    return commodities, months


def retrieve_monthly_summaries(commodities, months, vectorstore):
    context_parts = []
    for commodity in commodities:
        for month in months:
            filter = {
                "commodity": commodity,
                "month": month
            }
            retriever = vectorstore.as_retriever(
                search_kwargs={"k":1, "filter":filter}
            )
            result = retriever.invoke(f"{commodity} in {month}")
            if result:
                for doc in result:
                    part = f"Monthly Summary for {commodity} in {month}:\n{doc.page_content.strip()}\n"
                    context_parts.append(part)
    return "\n".join(context_parts)


def retrieve_world_news(months, vectorstore):
    context_parts = []
    for month in months:
        filter = {
            "month": month,
            "topic": "World News"
        }
        retriever = vectorstore.as_retriever(
            search_kwargs={"k":1, "filter":filter}
        )
        result = retriever.invoke(f"World news in {month}")
        if result:
            for doc in result:
                part = f"World News for {month}:\n{doc.page_content.strip()}\n"
                context_parts.append(part)
    return "\n".join(context_parts)


def retrieve_trading_strategies(commodities, vectorstore):
    context_parts = []
    for commodity in commodities:
        filter = {
            "commodity": commodity,
            "topic": "Trading Strategy"
        }
        retriever = vectorstore.as_retriever(
            search_kwargs={"k":1, "filter":filter}
        )
        result = retriever.invoke(f"Trading strategy for {commodity}")
        if result:
            for doc in result:
                part = f"Trading Strategy for {commodity}:\n{doc.page_content.strip()}\n"
                context_parts.append(part)
    return "\n".join(context_parts)


def main():
    print("\nReady. Ask me about commodities, world news or strategies (type 'exit' to quit):\n")

    while True:
        user_query = input("Your question: ")

        if user_query.lower() in ["exit", "quit"]:
            break

        commodities, months = extract_entities_from_query(user_query)

        if not commodities:
            commodities = ["Gold", "Crude Oil"]

        if not months:
            months = ["2025-04"]

        monthly_context = retrieve_monthly_summaries(commodities, months, commodity_vectorstore)
        news_context = retrieve_world_news(months, news_vectorstore)
        strategy_context = retrieve_trading_strategies(commodities, strategy_vectorstore)

        full_context = f"{monthly_context}\n\n{news_context}\n\n{strategy_context}"

        filled_prompt = query_template.invoke({
            'context': full_context,
            'question': user_query
        })

        print("\n--- Prompt to LLM ---")
        print(filled_prompt)

        response = model.invoke(filled_prompt)
        text = response.content.strip()

        answer = text.split("Answer:", 1)[-1].strip()

        answer = re.sub(r"^(</s>|<\|assistant\|>)*", "", answer).strip()

        print("\n--- Assistant's Answer ---")
        print(answer)

if __name__ == "__main__":
    main()

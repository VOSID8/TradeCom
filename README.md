# TradeCom: Commodity Trading RAG Pipeline
TradeCom is a Retrieval-Augmented Generation (RAG) system for commodity market analysis. It integrates monthly price summaries, world news, and trading strategies to provide evidence-backed answers to natural language queries.

## Features
- Integrates monthly price summaries, world news, and trading strategies.
- Allows the pipeline to run LLMs locally - this way, internal strategies of the firm don't get leaked outside.
- Allows natural language queries, such as:
- "Was May a good time to invest in Crude Oil?"
- Produces explainable, context-rich answers supported by evidence.

## Example Query
### User: "In march 2025 keeping Iran news in mind should we have invested in oil ?”
"Based on the provided context and world news for March 2025, it is recommended to ignore any investment decision regarding Crude Oil during this time period due to the ongoing conflicts and geopolitical risks in the Middle East, which include Iran and the NATO summit taking place in Washington D.C. Additionally, there is a high likelihood of supply and demand analysis as OPEC decisions, U.S. Shale production, and inventory data being monitored. Seasonality is also a consideration in this time period, as crude oil typically fluctuates in the summer driving season and weaker demand in spring/fall."

## Models used by me
Both are chosen keeping in mind my laptop configuration (I don’t have any dedicated GPU). all-MiniLM-L6-v2 is lightweight, fast, and accurate embedding model that captures semantic similarity well, ideal for efficient retrieval on local hardware.
For generation, TinyLlama-1.1B-Chat-v1 was selected as it’s a small, chat-tuned LLM that runs on CPU/GPU locally, producing coherent, grounded answers at low latency.

<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/6e57dc42-3cf6-43e9-9101-e405cfa75196" />

Medium blog - https://medium.com/@siddharthbanga/tradecom-commodity-trading-rag-pipeline-poc-075eb72aa422


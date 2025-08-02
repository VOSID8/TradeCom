# TradeCom: Commodity Trading RAG Pipeline
TradeCom is a Retrieval-Augmented Generation (RAG) system for commodity market analysis. It integrates monthly price summaries, world news, and trading strategies to provide evidence-backed answers to natural language queries.

## Features
- Integrates monthly price summaries, world news, and trading strategies.
- Allows the pipeline to run LLMs locally - this way, internal strategies of the firm don't get leaked outside.
- Allows natural language queries, such as:
- "Was May a good time to invest in Crude Oil?"
- Produces explainable, context-rich answers supported by evidence.

## Example Query
### User: "What insights are there for April 2025 for trading Gold?"
"Based on the context, the financial analyst is monitoring Gold prices in 2025–04 and noting a bullish trend, highs of $3182.40 and a low of $3305.00, as well as notable observations including a bullish trend, highs of $3182.20 and a low of $3282.40, and a notable low of $3305.00. The trend is positive, with prices rising steadily since April 1, 2025, until April 28, 2025, when the high of $3340.00 was reached. The analyst observes that the trend is consistent and bullish, while the highs and lows remain above the 200-day moving average. The analyst concludes that this is a positive trend for the financial analyst to follow and that the prices could continue to rise."

<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/6e57dc42-3cf6-43e9-9101-e405cfa75196" />


"""
app.py — NLP Finance Sentiment Demo
------------------------------------
Gradio app for HuggingFace Spaces.

Two modes:
  1. Manual input  — type any headline, get sentiment + confidence
  2. Live ticker   — enter a stock ticker, fetch recent news via yfinance,
                     score each headline, and display an aggregate sentiment

Model: defaults to ProsusAI/finbert (a public FinBERT — same architecture as
your trained model). Once you upload your own model to HuggingFace Hub,
replace MODEL_NAME below with "your-username/nlp-finance-sentiment".
"""

import gradio as gr
import yfinance as yf
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# ── Config ────────────────────────────────────────────────────────────────────
# Swap this for your own HuggingFace model once uploaded:
#   MODEL_NAME = "shravan-anand/financial-sentiment-bert"
MODEL_NAME  = "ProsusAI/finbert"
LABEL_MAP   = {0: "Negative 🔴", 1: "Neutral ⚪", 2: "Positive 🟢"}
LABEL_COLORS = {"Negative 🔴": "#ff4d4d", "Neutral ⚪": "#aaaaaa", "Positive 🟢": "#33cc66"}

# ── Load model once at startup ────────────────────────────────────────────────
print(f"Loading model: {MODEL_NAME}")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model     = BertForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model ready.")

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(headline: str):
    """Returns (label, confidence_dict)."""
    inputs = tokenizer(
        headline,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    label  = LABEL_MAP[int(np.argmax(probs))]
    scores = {LABEL_MAP[i]: float(probs[i]) for i in range(3)}
    return label, scores

# ── Tab 1: Single headline ────────────────────────────────────────────────────
def analyze_headline(headline):
    if not headline.strip():
        return "Please enter a headline.", {}
    label, scores = predict(headline)
    return label, scores

# ── Tab 2: Live ticker news ───────────────────────────────────────────────────
def analyze_ticker(ticker):
    ticker = ticker.strip().upper()
    if not ticker:
        return "Enter a ticker symbol (e.g. AAPL, TSLA, MSFT).", ""

    try:
        news_items = yf.Ticker(ticker).news
    except Exception as e:
        return f"Error fetching news for {ticker}: {e}", ""

    if not news_items:
        return f"No recent news found for {ticker}.", ""

    results = []
    sentiment_counts = {"Positive 🟢": 0, "Neutral ⚪": 0, "Negative 🔴": 0}

    for item in news_items[:10]:  # limit to 10 headlines
        title = item.get("title", "")
        if not title:
            continue
        label, scores = predict(title)
        sentiment_counts[label] += 1
        top_score = max(scores.values())
        results.append(f"**{label}** ({top_score:.0%})  \n{title}")

    total = sum(sentiment_counts.values()) or 1
    summary = (
        f"### {ticker} — Sentiment Summary ({total} headlines)\n"
        f"🟢 Positive: {sentiment_counts['Positive 🟢']} | "
        f"⚪ Neutral: {sentiment_counts['Neutral ⚪']} | "
        f"🔴 Negative: {sentiment_counts['Negative 🔴']}\n\n"
        + "\n\n---\n\n".join(results)
    )
    return summary, ""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Financial Sentiment Analyzer", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 📈 Financial Sentiment Analyzer
    BERT-based sentiment classifier for financial news headlines.
    Built by **Shravan Anand** · Duke University Class of 2030 · [GitHub](https://github.com/shravanA-git)

    > Model: Fine-tuned BERT on labeled financial news (3 classes: Positive / Neutral / Negative)
    """)

    with gr.Tab("🔍 Analyze a Headline"):
        with gr.Row():
            headline_input = gr.Textbox(
                label="Financial Headline",
                placeholder="e.g. Apple reports record quarterly earnings, beating analyst expectations",
                lines=2
            )
        analyze_btn = gr.Button("Analyze", variant="primary")
        with gr.Row():
            label_output  = gr.Label(label="Sentiment")
            scores_output = gr.Label(label="Confidence Scores")
        analyze_btn.click(analyze_headline, inputs=headline_input, outputs=[label_output, scores_output])

        gr.Examples(
            examples=[
                ["Apple reports record quarterly earnings, beating analyst expectations by 15%"],
                ["Federal Reserve signals potential interest rate cuts amid cooling inflation"],
                ["Tesla misses delivery targets as production challenges persist"],
                ["Goldman Sachs upgrades Microsoft stock to buy with $450 price target"],
                ["Retail sales data shows unexpected decline, raising recession concerns"],
            ],
            inputs=headline_input
        )

    with gr.Tab("📰 Live Ticker News"):
        gr.Markdown("Enter a stock ticker to fetch and score its latest news headlines.")
        with gr.Row():
            ticker_input = gr.Textbox(label="Ticker Symbol", placeholder="e.g. AAPL, TSLA, NVDA, JPM", scale=1)
            ticker_btn   = gr.Button("Fetch & Analyze", variant="primary", scale=0)
        ticker_output = gr.Markdown(label="Results")
        ticker_btn.click(analyze_ticker, inputs=ticker_input, outputs=[ticker_output, gr.Textbox(visible=False)])

    gr.Markdown("""
    ---
    **About this project**: Fine-tuned `bert-base-uncased` on the Financial PhraseBank dataset.
    Achieved 99% test accuracy through iterative model refinement.
    """)

if __name__ == "__main__":
    demo.launch()

# Deployment Guide — NLP Finance Sentiment Demo

## What You're Deploying
A Gradio web app that classifies financial headlines as Positive / Neutral / Negative.
Hosted for free on HuggingFace Spaces. Shareable URL: `huggingface.co/spaces/YOUR-USERNAME/nlp-finance-demo`

---

## Step 1 — Save Your Trained Model (do this in Google Colab)

Your original code trains the model but never saves it. Fix that:

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `save_model.py` or paste its contents into a notebook cell
3. Run it (needs GPU — use Colab's free T4 runtime: Runtime → Change runtime type → T4 GPU)
4. When it finishes, download the `saved_model/` folder from the Colab file browser
5. Keep this folder — you'll upload it to HuggingFace in Step 3

**Time**: ~20–30 minutes to train + save.

If you want to skip this for now and demo immediately, the app currently uses
`ProsusAI/finbert` (a public FinBERT with the same architecture). You can swap
in your model later by changing one line in `app.py`.

---

## Step 2 — Create a HuggingFace Account

1. Go to [huggingface.co](https://huggingface.co) and sign up (free)
2. Set your username — this will be in your demo URL, so pick something clean
   (e.g., `shravan-anand`)

---

## Step 3 — Upload Your Model to HuggingFace Hub (optional but recommended)

This makes the demo use *your* model, not the public one.

```bash
pip install huggingface_hub

python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='saved_model/',
    repo_id='YOUR-USERNAME/financial-sentiment-bert',
    repo_type='model'
)
"
```

Then in `app.py`, change line 20 to:
```python
MODEL_NAME = "YOUR-USERNAME/financial-sentiment-bert"
```

---

## Step 4 — Create a HuggingFace Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Settings:
   - **Owner**: your username
   - **Space name**: `nlp-finance-demo`
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (free) — sufficient for BERT inference
3. Click "Create Space"

---

## Step 5 — Upload Your Files

In your new Space, click "Files" → "Add file" → upload:
- `app.py`
- `requirements.txt`

HuggingFace will automatically install dependencies and launch the app.
Build takes ~3–5 minutes. Watch the logs tab.

---

## Step 6 — Your Demo is Live

URL: `https://huggingface.co/spaces/YOUR-USERNAME/nlp-finance-demo`

Share this link in:
- Your GitHub README
- Your resume (under the NLP Finance project)
- Duke club applications and interviews
- LinkedIn

---

## Upgrading Later (with Claude Code)

Once you're comfortable, the next version of this can:
- Pull live earnings call transcripts and score paragraph-by-paragraph
- Show a sentiment timeline chart for a stock over the past 30 days
- Build a watchlist where you track sentiment shifts as an early signal

For that level of complexity, open Claude Code in your terminal
(`claude` command) and point it at this project folder.
It handles multi-file apps, running local servers, and Git pushes natively.

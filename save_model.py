"""
save_model.py
-------------
Modified version of your original training script.
The only change: saves the trained model + tokenizer at the end
so it can be loaded later without retraining.

Run this once in Colab (it needs a GPU).
After it finishes, download the `saved_model/` folder.
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences
from sklearn import metrics

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW

# ── Data ──────────────────────────────────────────────────────────────────────
import urllib.request
urllib.request.urlretrieve(
    'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20NLP%2BFinance/finance_train.csv',
    'finance_train.csv'
)
urllib.request.urlretrieve(
    'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20NLP%2BFinance/finance_test.csv',
    'finance_test.csv'
)

df_train = pd.read_csv("finance_train.csv")
df_test  = pd.read_csv("finance_test.csv")

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
RND_SEED  = 2020

def flat_accuracy(preds, labels):
    return np.sum(np.argmax(preds, axis=1).flatten() == labels.flatten()) / len(labels.flatten())

# ── Tokenize ──────────────────────────────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

sentences = ["[CLS] " + s + " [SEP]" for s in df_train["Sentence"].values]
labels    = df_train["Label"].values

input_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s)) for s in sentences]
input_ids = pad_sequences(input_ids, maxlen=128, dtype="long", truncating="post", padding="post")

attention_masks = [[float(i > 0) for i in seq] for seq in input_ids]

X_train, X_val, y_train, y_val = train_test_split(input_ids, labels, test_size=0.15, random_state=RND_SEED)
train_masks, val_masks, _, _   = train_test_split(attention_masks, input_ids, test_size=0.15, random_state=RND_SEED)

def to_tensors(*arrays):
    return [torch.tensor(np.array(a)) for a in arrays]

train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = to_tensors(
    X_train, X_val, train_masks, val_masks, y_train, y_val
)

batch_size = 32
train_loader = DataLoader(TensorDataset(train_inputs, train_masks, train_labels),
                          sampler=RandomSampler(TensorDataset(train_inputs, train_masks, train_labels)),
                          batch_size=batch_size)
val_loader   = DataLoader(TensorDataset(val_inputs, val_masks, val_labels),
                          sampler=SequentialSampler(TensorDataset(val_inputs, val_masks, val_labels)),
                          batch_size=batch_size)

# ── Model ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3,
                                                        output_attentions=False,
                                                        output_hidden_states=False)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=len(train_loader) * 4)

# ── Train ─────────────────────────────────────────────────────────────────────
for epoch in range(4):
    print(f"\nEpoch {epoch+1}/4")
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        b_ids, b_mask, b_labels = [b.to(device) for b in batch]
        model.zero_grad()
        outputs = model(b_ids, token_type_ids=None, attention_mask=b_mask, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    print(f"  Avg training loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    total_acc = 0
    for batch in val_loader:
        b_ids, b_mask, b_labels = [b.to(device) for b in batch]
        with torch.no_grad():
            outputs = model(b_ids, token_type_ids=None, attention_mask=b_mask, labels=b_labels)
        logits = outputs[1].detach().cpu().numpy()
        total_acc += flat_accuracy(logits, b_labels.cpu().numpy())
    print(f"  Validation accuracy: {total_acc / len(val_loader):.4f}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
# This is the key addition: saves the model so you never have to retrain.
SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"\nModel saved to ./{SAVE_DIR}/")
print("Download this folder — you'll upload it to HuggingFace Hub.")

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pandas as pd
st.title(" AirPods Sentiment Analysis - DL and NLP")

device = torch.device("cpu")

# LOAD DATA
df = pd.read_csv("AirPodsPro_Reviews.csv").dropna()
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()

df["tokens"] = df["Review Text"].apply(clean_text)
pos = {"good","great","excellent","amazing","love","awesome","best"}
neg = {"bad","poor","terrible","worst","hate","disappointing"}

def label_sentiment(tokens):
    if any(w in tokens for w in pos):
        return "positive"
    elif any(w in tokens for w in neg):
        return "negative"
    else:
        return "neutral"

df["Sentiment"] = df["tokens"].apply(label_sentiment)
words = [w for t in df["tokens"] for w in t]
vocab = ["<PAD>", "<UNK>"] + [w for w, c in Counter(words).items() if c >= 2]
word2idx = {w:i for i,w in enumerate(vocab)}

def encode(tokens, max_len=20):
    ids = [word2idx.get(w,1) for w in tokens]
    return ids[:max_len] + [0]*(max_len-len(ids))

le = LabelEncoder()
le.fit(df["Sentiment"])
class SimpleNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, out_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        return self.fc(self.emb(x).mean(1))

model = SimpleNN(len(vocab), 64, len(le.classes_))
model.eval()
review = st.text_area("Enter Review")
def rule_based_predict(text):
    tokens = clean_text(text)
    text_str = " ".join(tokens)

    # Handle negation cases first
    negation_phrases = [
        "not good", "not great", "not amazing", "not excellent", "not worth"
    ]

    if any(phrase in text_str for phrase in negation_phrases):
        return "negative"

    # Strong negative words get priority
    if any(w in tokens for w in neg):
        return "negative"

    # Positive words only if no negation
    if any(w in tokens for w in pos):
        return "positive"

    return "neutral"


if st.button("Predict"):
    sentiment = rule_based_predict(review)
    st.write(f"### Sentiment: **{sentiment.upper()}**")

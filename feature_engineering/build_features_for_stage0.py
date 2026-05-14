

import os
import re
import sys
import time
import warnings
import numpy as np
import pandas as pd
import nltk
import textstat
import torch
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")

REPO = "/Users/minsucho/Documents/Helpfulness/revisions"
CLEAN = os.path.join(REPO, "data", "cleaned_data")
LATENT = os.path.join(REPO, "data", "latent_data")
OUT = os.path.join(REPO, "data", "robustness", "features_baseline")
VA_PATH = os.path.join(REPO, "feature_engineering", "emotion_va_scores.csv")
os.makedirs(OUT, exist_ok=True)

try:
    STOP = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP = set(stopwords.words("english"))

MODEL_ID = "SamLowe/roberta-base-go_emotions"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH = 64
MAX_LEN = 512

PLATFORM_CFG = {
    "amazon":   {"text": "Review_Text", "title": "Review_Title", "rating": "Rating", "avg_rating": "Average_Rating", "recency": "Time_Lapsed"},
    "audible":  {"text": "Review_Text", "title": "Review_Title", "rating": "Rating", "avg_rating": "Average_Rating", "recency": "Time_Lapsed"},
    "coursera": {"text": "Review_Text", "title": None,           "rating": "Rating", "avg_rating": "Average_Rating", "recency": "Time_Lapsed"},
    "hotel":    {"text": "Review_Text", "title": "Review_Title", "rating": "Rating", "avg_rating": "Average_Rating", "recency": "Time_Lapsed"},
}


def compute_depth(text):
    tokens = re.findall(r"\b[a-z]+\b", (str(text) or "").lower())
    return sum(1 for w in tokens if w not in STOP)


def compute_breadth_nmf(texts, n_topics=10):
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(texts)
    nmf = NMF(n_components=n_topics, random_state=42, init="nndsvd")
    W = nmf.fit_transform(X)
    P = W / np.clip(W.sum(axis=1, keepdims=True), 1e-12, None)
    q = P.mean(axis=0)
    ratio = np.clip(P / q, 1e-10, None) + 1e-10
    return (P * np.log10(ratio)).sum(axis=1)


def build_arousal_emotion_arr(va_path):
    va = pd.read_csv(va_path)
    va_map = dict(zip(va["emotion"], va["arousal"]))
    return va_map


def batched_arousal(texts, tokenizer, model, va_map, label_ids):
    """Return a numpy float array of arousal scores aligned with texts.

    Each text is scored as a confidence-weighted average of the emotion-arousal
    values over the labels listed in `va_map`. Confidence is the sigmoid output
    of the multi-label classifier.
    """
    n = len(texts)
    arousal = np.full(n, np.nan, dtype=float)
    model.eval()
    # Precompute the arousal vector indexed by model-label order
    label_arousal = np.array([va_map.get(label, np.nan) for label in label_ids])
    valid_mask = ~np.isnan(label_arousal)
    label_arousal_valid = label_arousal[valid_mask]
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, BATCH):
            batch_texts = [str(t) if pd.notna(t) and str(t).strip() else "" for t in texts[start:start + BATCH]]
            empty = [i for i, t in enumerate(batch_texts) if t == ""]
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            logits = model(**enc).logits
            probs = torch.sigmoid(logits).detach().cpu().numpy()  # (B, L) multi-label
            valid = probs[:, valid_mask]
            weights_sum = valid.sum(axis=1)
            num = (valid * label_arousal_valid).sum(axis=1)
            scores = np.where(weights_sum > 0, num / np.maximum(weights_sum, 1e-12), np.nan)
            for i in empty:
                scores[i] = np.nan
            arousal[start:start + BATCH] = scores
            if (start // BATCH) % 20 == 0:
                elapsed = time.time() - t0
                rate = (start + BATCH) / max(elapsed, 1e-6)
                eta = (n - (start + BATCH)) / max(rate, 1e-6)
                print(f"  arousal {start + BATCH}/{n}  rate={rate:.1f}/s  eta={eta/60:.1f}min", flush=True)
    return arousal


def main():
    print(f"Device: {DEVICE}")
    print("Loading RoBERTa go_emotions...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(DEVICE)
    id2label = model.config.id2label
    label_ids = [id2label[i] for i in range(len(id2label))]
    va_map = build_arousal_emotion_arr(VA_PATH)
    missing_labels = [l for l in label_ids if l not in va_map]
    if missing_labels:
        print(f"  WARNING: {len(missing_labels)} emotion labels missing from VA map: {missing_labels}", flush=True)

    for platform, cfg in PLATFORM_CFG.items():
        out_path = os.path.join(OUT, f"{platform}.csv")
        if os.path.exists(out_path):
            print(f"[{platform}] already exists at {out_path}, skipping.", flush=True)
            continue
        print(f"\n=== {platform} ===", flush=True)
        df_clean = pd.read_csv(os.path.join(CLEAN, f"{platform}.csv"))
        df_latent = pd.read_csv(os.path.join(LATENT, f"{platform}.csv"))
        assert len(df_clean) == len(df_latent), f"row mismatch for {platform}"
        text_col = cfg["text"]

        out = pd.DataFrame(index=df_clean.index)
        out["Helpfulness"] = df_latent["Helpfulness"].values
        out["Group"] = df_latent["Group"].values

        t0 = time.time()
        print("  depth ...", flush=True)
        out["depth"] = df_clean[text_col].apply(compute_depth).astype(float)
        print(f"    done ({time.time() - t0:.1f}s)", flush=True)

        t0 = time.time()
        print("  breadth (NMF K=10) ...", flush=True)
        texts = df_clean[text_col].fillna("").astype(str).values
        out["breadth"] = compute_breadth_nmf(texts, n_topics=10)
        print(f"    done ({time.time() - t0:.1f}s)", flush=True)

        t0 = time.time()
        print("  readability (FRE) ...", flush=True)
        out["readability"] = df_clean[text_col].fillna("").astype(str).apply(textstat.flesch_reading_ease)
        print(f"    done ({time.time() - t0:.1f}s)", flush=True)

        t0 = time.time()
        print("  arousal (RoBERTa) ...", flush=True)
        out["arousal"] = batched_arousal(texts, tokenizer, model, va_map, label_ids)
        print(f"    done ({time.time() - t0:.1f}s)", flush=True)

        out["rating_deviation"] = (df_clean[cfg["rating"]] - df_clean[cfg["avg_rating"]]).abs()

        if cfg["title"] is not None:
            out["title_length"] = df_clean[cfg["title"]].fillna("").astype(str).str.len()
        else:
            out["title_length"] = np.nan

        out["recency"] = df_clean[cfg["recency"]].astype(float)

        out.to_csv(out_path, index=False)
        print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

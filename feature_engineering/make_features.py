import re
import os
import pandas as pd
import numpy as np
import torch
import nltk
import textstat
from typing import List
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Setup NLTK Stopwords
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))

# Load RoBERTa Model for Arousal Extraction
MODEL_ID = "SamLowe/roberta-base-go_emotions"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

def compute_depth(text: str) -> int:
    tokens = re.findall(r"\b[a-z]+\b", (str(text) or "").lower())
    return sum(1 for w in tokens if w not in STOP_WORDS)

def compute_breadth_nmf(df, text_col, n_topics=10):
    texts = df[text_col].fillna("").astype(str).values
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(texts)
    
    nmf = NMF(n_components=n_topics, random_state=42, init='nndsvd')
    W = nmf.fit_transform(X)
    
    P = W / np.clip(W.sum(axis=1, keepdims=True), 1e-12, None)
    q = P.mean(axis=0)
    
    # Calculate Breadth based on KL Divergence
    breadth = [np.sum(P[i] * np.log10(np.clip(P[i]/q, 1e-10, None) + 1e-10)) for i in range(P.shape[0])]
    return pd.Series(breadth, index=df.index)

def calculate_arousal(emotion_results, va_df):
    if not emotion_results: return None
    weighted_arousal, total_weight = 0, 0
    for emotion in emotion_results:
        label, score = emotion['label'], emotion['score']
        va_row = va_df[va_df['emotion'] == label]
        if not va_row.empty:
            arousal_val = va_row['arousal'].iloc[0]
            if not pd.isna(arousal_val):
                weighted_arousal += arousal_val * score
                total_weight += score
    return weighted_arousal / total_weight if total_weight > 0 else None

def build_all_features(df, va_scores_path, text_col='Review_Text', title_col='Review_Title', 
                       rating_col='Rating', avg_rating_col='Average_Rating', 
                       posted_col='Posted_Date', crawled_date='24-12-05'):
    
    print("Starting Feature Engineering Pipeline...")
    va_df = pd.read_csv(va_scores_path)
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, 
                          device=device, top_k=None, function_to_apply="sigmoid", truncation=True)

    # 1. Systematic Cues & Text-based Heuristics
    if text_col in df.columns:
        print(f"- Processing Systematic Cues (Depth, Breadth, FRE, Arousal) using '{text_col}'...")
        df['Depth'] = df[text_col].apply(compute_depth)
        df['FRE'] = df[text_col].apply(lambda x: textstat.flesch_reading_ease(str(x)))
        df['Breadth'] = compute_breadth_nmf(df, text_col)
        
        arousal_list = []
        for text in df[text_col]:
            if pd.isna(text) or str(text).strip() == "":
                arousal_list.append(None)
                continue
            res = classifier(str(text)[:512]) 
            arousal_list.append(calculate_arousal(res[0], va_df))
        df['Arousal'] = arousal_list
        df['ReviewLength'] = df[text_col].fillna("").str.len()
    else:
        print(f"Column '{text_col}' not found. Skipping Systematic Cues and ReviewLength.")

    # 2. Title Length
    if title_col in df.columns:
        print(f"- Processing TitleLength using '{title_col}'...")
        df['TitleLength'] = df[title_col].fillna("").str.len()
    else:
        print(f"Column '{title_col}' not found. Skipping TitleLength.")

    # 3. Rating Deviation
    if rating_col in df.columns and avg_rating_col in df.columns:
        print(f"- Processing RatingDeviation using '{rating_col}' and '{avg_rating_col}'...")
        df['RatingDeviation'] = (df[rating_col] - df[avg_rating_col]).abs()
    else:
        print(f"Column '{rating_col}' or '{avg_rating_col}' not found. Skipping RatingDeviation.")

	# 4. Time Lapsed (Recency)
    if posted_col in df.columns:
        print(f"- Processing Time_Lapsed (Recency) using '{posted_col}'...")
        
        posted = pd.to_datetime(df[posted_col], errors='coerce')
        
        crawled = pd.to_datetime(crawled_date, format="%y-%m-%d")
        
        df['Time_Lapsed'] = (crawled - posted).dt.days
    else:
        print(f"Column '{posted_col}' not found. Skipping Time_Lapsed.")

    print("Pipeline execution completed.")
    return df

if __name__ == "__main__":
    
    """
    How to use:
    1. Set 'RAW_DATA_PATH' and 'OUTPUT_PATH'.
    2. Ensure 'VA_SCORES_PATH' points to your emotion-arousal mapping file.
    3. Run this script to generate all cognitive processing features.
    """
    
    RAW_DATA_PATH = "../data/cleaned_data/audible_sample.csv"
    VA_SCORES_PATH = "./emotion_va_scores.csv" # Emotion mapping file
    OUTPUT_PATH = "../data/processed_audible_sample.csv"
    
    if os.path.exists(RAW_DATA_PATH):
        df_raw = pd.read_csv(RAW_DATA_PATH)
        df_final = build_all_features(df_raw, VA_SCORES_PATH)
        df_final.to_csv(OUTPUT_PATH, index=False)
    else:
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}")
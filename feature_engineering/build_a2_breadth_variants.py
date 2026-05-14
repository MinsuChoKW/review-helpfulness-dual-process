

import os
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

warnings.filterwarnings("ignore")

REPO = "/Users/minsucho/Documents/Helpfulness/revisions"
CLEAN_DIR = os.path.join(REPO, "data", "cleaned_data")
FEAT_DIR  = os.path.join(REPO, "data", "robustness", "features_baseline")
OUT_BASE  = os.path.join(REPO, "data", "robustness", "smartpls_input")

PLATFORMS = ["amazon", "audible", "coursera", "hotel"]
TEXT_COL  = "Review_Text"  # same in all 4 cleaned data files

# Header order in the output CSVs. Coursera will skip TitleLength.
SYS_COLS_OUT = ["Depth", "Breadth", "Readability", "Arousal"]
HEU_COLS_OUT_FULL  = ["RatingDeviation", "TitleLength", "Recency"]
HEU_COLS_OUT_COURS = ["RatingDeviation", "Recency"]

# Map our internal lowercase indicator names to the output column names.
RENAME = {
    "depth": "Depth",
    "breadth": "Breadth",
    "readability": "Readability",
    "arousal": "Arousal",
    "rating_deviation": "RatingDeviation",
    "title_length": "TitleLength",
    "recency": "Recency",
}


def fit_nmf_topic_mixtures(texts, n_topics):
    """Reproduce the baseline NMF pipeline: TfidfVectorizer + NMF with the
    exact baseline settings, varying only n_components."""
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(texts)
    nmf = NMF(n_components=n_topics, random_state=42, init="nndsvd")
    W = nmf.fit_transform(X)
    P = W / np.clip(W.sum(axis=1, keepdims=True), 1e-12, None)
    return P


def breadth_kl(P):
    """Base-10 KL divergence vs. corpus mean topic mixture (matches baseline make_features.py)."""
    q = P.mean(axis=0)
    ratio = np.clip(P / q, 1e-10, None) + 1e-10
    return (P * np.log10(ratio)).sum(axis=1)


def breadth_entropy(P):
    """Topic entropy −Σ p log10 p (base 10 for comparability with the KL baseline)."""
    Pc = np.clip(P, 1e-12, None)
    return -(Pc * np.log10(Pc)).sum(axis=1)


def write_smartpls_csv(out_path, df_features, heu_cols, sys_cols):
    cols = sys_cols + heu_cols + ["Helpfulness", "Group"]
    sub = df_features[cols].copy()
    sub.to_csv(out_path, index=False)


def main():
    for platform in PLATFORMS:
        t0 = time.time()
        print(f"\n=== {platform} ===", flush=True)
        baseline = pd.read_csv(os.path.join(FEAT_DIR, f"{platform}.csv"))
        clean = pd.read_csv(os.path.join(CLEAN_DIR, f"{platform}.csv"))
        assert len(baseline) == len(clean), f"row count mismatch for {platform}"
        texts = clean[TEXT_COL].fillna("").astype(str).values

        # Build a master feature table on this platform that uses the
        # manuscript-style column names.
        master = baseline.rename(columns=RENAME).copy()
        is_coursera = (platform == "coursera")
        heu_out = HEU_COLS_OUT_COURS if is_coursera else HEU_COLS_OUT_FULL

        # ---- A2-a: NMF K=5 → KL ----
        t = time.time()
        P5 = fit_nmf_topic_mixtures(texts, n_topics=5)
        a2a = master.copy()
        a2a["Breadth"] = breadth_kl(P5)
        out_path = os.path.join(OUT_BASE, "A2-a", f"{platform}.csv")
        write_smartpls_csv(out_path, a2a, heu_out, SYS_COLS_OUT)
        print(f"  A2-a (NMF K=5)  -> {out_path}   ({time.time()-t:.1f}s)", flush=True)

        # ---- A2-b: NMF K=15 → KL ----
        t = time.time()
        P15 = fit_nmf_topic_mixtures(texts, n_topics=15)
        a2b = master.copy()
        a2b["Breadth"] = breadth_kl(P15)
        out_path = os.path.join(OUT_BASE, "A2-b", f"{platform}.csv")
        write_smartpls_csv(out_path, a2b, heu_out, SYS_COLS_OUT)
        print(f"  A2-b (NMF K=15) -> {out_path}   ({time.time()-t:.1f}s)", flush=True)

        # ---- A2-c: entropy on K=10 mixtures ----
        t = time.time()
        P10 = fit_nmf_topic_mixtures(texts, n_topics=10)
        a2c = master.copy()
        a2c["Breadth"] = breadth_entropy(P10)
        out_path = os.path.join(OUT_BASE, "A2-c", f"{platform}.csv")
        write_smartpls_csv(out_path, a2c, heu_out, SYS_COLS_OUT)
        print(f"  A2-c (entropy K=10) -> {out_path}   ({time.time()-t:.1f}s)", flush=True)

        # Sanity check: the K=10 KL recomputed here should equal master.Breadth
        # (the baseline breadth that fed Stage 0 SmartPLS). Reported once per platform.
        kl_recomputed = breadth_kl(P10)
        baseline_kl   = master["Breadth"].values
        diff = np.max(np.abs(kl_recomputed - baseline_kl))
        print(f"  sanity: max |K=10 KL recompute - baseline Breadth| = {diff:.3e}", flush=True)

        # ---- D1: drop Recency from CSV ----
        t = time.time()
        d1_heu = [c for c in heu_out if c != "Recency"]
        out_path = os.path.join(OUT_BASE, "D1", f"{platform}.csv")
        write_smartpls_csv(out_path, master, d1_heu, SYS_COLS_OUT)
        print(f"  D1 (no recency)  -> {out_path}   ({time.time()-t:.1f}s)", flush=True)

        print(f"  total: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

"""
Stage 0: reproduce the SmartPLS 4 baseline indicator weights using the
Python port of `plspm` (the R package's official port).

Input:  data/robustness/features_baseline/<platform>.csv  (seven indicators)
Output: results/robustness/stage0_plspm_vs_smartpls.csv   (long format)

Why Python plspm, not R plspm?
- R is not installed on this machine; installing the full R toolchain is heavy.
- The Python package is the official port of the R `plspm` package and exposes
  the same model: formative blocks via Mode.B, inner weighting via Scheme.PATH,
  bootstrap via `bootstrap=True, bootstrap_iterations=1000`.

Model (per `docs/baseline_specification.md`):
  - Heuristic  (Mode.B): rating_deviation, title_length, recency
                         (coursera drops title_length)
  - Systematic (Mode.B): depth, breadth, readability, arousal
  - Outcome    (Mode.A, single MV): Helpfulness
  - Paths: Heuristic -> Outcome, Systematic -> Outcome
  - Scheme: PATH (matches SmartPLS 4 default; differs from plspm's centroid default)
  - Bootstrap: 1000 resamples for indicator-weight p-values
  - Scaled: True (plspm default; matches SmartPLS standardization)

The Outcome block exists only to satisfy plspm's requirement of at least one path
in the inner model. It plays no role in the outer (indicator) weights of the
formative blocks; we read those off `outer_model()` and `bootstrap().weights()`.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

from plspm.config import Config, MV, Structure
from plspm.plspm import Plspm
from plspm.mode import Mode
from plspm.scheme import Scheme
from scipy.stats import norm

warnings.filterwarnings("ignore")

REPO = "/Users/minsucho/Documents/Helpfulness/revisions"
FEAT = os.path.join(REPO, "data", "robustness", "features_baseline")
OUT_CSV = os.path.join(REPO, "results", "robustness", "stage0_plspm_vs_smartpls.csv")
LOG_DIR = os.path.join(REPO, "results", "robustness")
os.makedirs(LOG_DIR, exist_ok=True)

PLATFORMS = ["amazon", "audible", "coursera", "hotel"]
SYS_INDICATORS = ["depth", "breadth", "readability", "arousal"]
HEU_INDICATORS_FULL = ["rating_deviation", "title_length", "recency"]
HEU_INDICATORS_COURSERA = ["rating_deviation", "recency"]  # no title_length

BOOTSTRAP_ITERS = 1000
PROCESSES = 4  # bootstrap_iterations must be a multiple of processes

# SmartPLS 4 reference values from the manuscript's Tables 2 and 3.
# Significance encoding: "***" p<0.001, "**" p<0.01, "*" p<0.05, "n.s." not sig.
# Platform name "hotel" corresponds to "Booking.com" in the manuscript.
SMARTPLS_REF = {
    # (platform, indicator) : (weight, sig)
    ("amazon",   "depth"):            (1.026, "***"),
    ("amazon",   "breadth"):          (-0.085, "***"),
    ("amazon",   "readability"):      (0.044, "***"),
    ("amazon",   "arousal"):          (-0.026, "***"),
    ("amazon",   "rating_deviation"): (0.574, "***"),
    ("amazon",   "title_length"):     (0.736, "***"),
    ("amazon",   "recency"):          (0.125, "n.s."),

    ("hotel",    "depth"):            (1.040, "***"),
    ("hotel",    "breadth"):          (0.251, "***"),
    ("hotel",    "readability"):      (0.012, "n.s."),
    ("hotel",    "arousal"):          (0.085, "***"),
    ("hotel",    "rating_deviation"): (0.919, "***"),
    ("hotel",    "title_length"):     (0.390, "***"),
    ("hotel",    "recency"):          (0.139, "***"),

    ("audible",  "depth"):            (1.028, "***"),
    ("audible",  "breadth"):          (-0.108, "***"),
    ("audible",  "readability"):      (0.048, "***"),
    ("audible",  "arousal"):          (-0.034, "n.s."),
    ("audible",  "rating_deviation"): (0.582, "***"),
    ("audible",  "title_length"):     (0.142, "**"),
    ("audible",  "recency"):          (0.755, "***"),

    ("coursera", "depth"):            (1.024, "***"),
    ("coursera", "breadth"):          (-0.159, "***"),
    ("coursera", "readability"):      (0.042, "**"),
    ("coursera", "arousal"):          (-0.063, "***"),
    ("coursera", "rating_deviation"): (0.986, "***"),
    # title_length: NA for coursera
    ("coursera", "recency"):          (0.227, "**"),
}

# Significance rank order for "matches if within ±1 category" tolerance comparisons.
SIG_RANK = {"n.s.": 0, "*": 1, "**": 2, "***": 3, "": np.nan}


def build_config(heu_indicators):
    """Important: scaled=False here because the Python plspm port's "scaling"
    code divides every column by the POOLED standard deviation
    (`metric_data.stack().std()`), not by each column's own std. That mangles
    formative weights when indicators have very different raw scales (e.g.,
    `recency` in days vs. `arousal` in [0,1]). We z-score each column
    ourselves in `run_platform()` and pass `scaled=False` here. This
    reproduces what SmartPLS / R plspm do per-column."""
    structure = Structure()
    structure.add_path(source=["Heuristic", "Systematic"], target=["Outcome"])
    config = Config(structure.path(), scaled=False)
    config.add_lv("Heuristic", Mode.B, *(MV(i) for i in heu_indicators))
    config.add_lv("Systematic", Mode.B, *(MV(i) for i in SYS_INDICATORS))
    config.add_lv("Outcome", Mode.A, MV("Helpfulness"))
    return config


def zscore_columns(df, columns):
    """Standardize specified columns to mean 0, std 1 (sample std, ddof=1)."""
    out = df.copy()
    for c in columns:
        col = out[c].astype(float)
        out[c] = (col - col.mean()) / col.std(ddof=1)
    return out


def stars(p):
    if p is None or (isinstance(p, float) and (np.isnan(p) or p > 1)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def run_platform(platform):
    feat_path = os.path.join(FEAT, f"{platform}.csv")
    print(f"\n=== {platform} ===  reading {feat_path}", flush=True)
    df = pd.read_csv(feat_path)
    heu = HEU_INDICATORS_COURSERA if platform == "coursera" else HEU_INDICATORS_FULL
    needed = ["Helpfulness"] + heu + SYS_INDICATORS
    df = df[needed].copy()
    n_pre = len(df)
    df = df.dropna()
    n_post = len(df)
    print(f"  rows: {n_pre} -> {n_post} after dropping NaN indicators", flush=True)
    df = zscore_columns(df, needed)

    config = build_config(heu)
    t0 = time.time()
    p = Plspm(
        data=df,
        config=config,
        scheme=Scheme.PATH,
        bootstrap=True,
        bootstrap_iterations=BOOTSTRAP_ITERS,
        processes=PROCESSES,
    )
    print(f"  plspm fit: {time.time() - t0:.1f}s", flush=True)

    outer = p.outer_model()
    boot_w = p.bootstrap().weights()
    paths = p.path_coefficients()
    print("  outer_model:", flush=True)
    print(outer.round(4), flush=True)
    print("  bootstrap weights:", flush=True)
    print(boot_w.round(4), flush=True)
    print("  inner paths:", flush=True)
    print(paths.round(4), flush=True)

    rows = []
    for ind in heu:
        rows.append({"construct": "Heuristic", "indicator": ind})
    for ind in SYS_INDICATORS:
        rows.append({"construct": "Systematic", "indicator": ind})

    out_rows = []
    for r in rows:
        ind = r["indicator"]
        weight = outer.loc[ind, "weight"] if ind in outer.index else np.nan
        if ind in boot_w.index:
            row_boot = boot_w.loc[ind]
            boot_orig = row_boot["original"]
            boot_se = row_boot["std.error"]
            ci_lo = row_boot["perc.025"]
            ci_hi = row_boot["perc.975"]
            tstat = row_boot["t stat."]
        else:
            boot_orig = weight
            boot_se = np.nan
            ci_lo = np.nan
            ci_hi = np.nan
            tstat = np.nan
        if pd.notna(tstat):
            p_val = 2 * (1 - norm.cdf(abs(tstat)))
        else:
            p_val = np.nan

        ref = SMARTPLS_REF.get((platform, ind))
        if ref is not None:
            ref_weight, ref_sig = ref
        else:
            ref_weight, ref_sig = (np.nan, "")
        plspm_sig = stars(p_val)
        abs_diff = abs(float(weight) - ref_weight) if pd.notna(weight) and pd.notna(ref_weight) else np.nan
        if pd.notna(weight) and pd.notna(ref_weight):
            sign_match = int(np.sign(weight) == np.sign(ref_weight) or (weight == 0 and ref_weight == 0))
        else:
            sign_match = np.nan
        if plspm_sig and ref_sig:
            sig_match = int(plspm_sig == ref_sig)
        else:
            sig_match = np.nan

        out_rows.append({
            "platform": platform,
            "construct": r["construct"],
            "indicator": ind,
            "n": n_post,
            "plspm_weight": float(weight) if pd.notna(weight) else np.nan,
            "plspm_boot_mean": float(row_boot["mean"]) if ind in boot_w.index else np.nan,
            "plspm_boot_se": float(boot_se) if pd.notna(boot_se) else np.nan,
            "plspm_t_stat": float(tstat) if pd.notna(tstat) else np.nan,
            "plspm_ci_lo": float(ci_lo) if pd.notna(ci_lo) else np.nan,
            "plspm_ci_hi": float(ci_hi) if pd.notna(ci_hi) else np.nan,
            "plspm_p_value": float(p_val) if pd.notna(p_val) else np.nan,
            "plspm_sig": plspm_sig,
            "smartpls_weight": ref_weight,
            "smartpls_sig": ref_sig,
            "abs_diff": abs_diff,
            "sign_match": sign_match,
            "sig_match": sig_match,
        })
    return pd.DataFrame(out_rows)


def assess(full):
    """Compute and print the three success-criteria checks against the
    27-cell threshold (4 platforms x 7 indicators - 1 NA for coursera title)."""
    cells_total = len(full)
    weight_tol = 0.03
    within_tol = (full["abs_diff"] <= weight_tol).sum()
    sign_ok = (full["sign_match"] == 1).sum()
    sig_ok = (full["sig_match"] == 1).sum()
    print("\n--- Stage 0 success-criteria assessment ---", flush=True)
    print(f"  cells total: {cells_total}", flush=True)
    print(f"  |plspm - smartpls| <= {weight_tol}: {within_tol}/{cells_total}", flush=True)
    print(f"  sign match:                       {sign_ok}/{cells_total}", flush=True)
    print(f"  significance category match:      {sig_ok}/{cells_total} (threshold: 25)", flush=True)
    failures = full.loc[(full["abs_diff"] > weight_tol) | (full["sign_match"] == 0) | (full["sig_match"] == 0),
                        ["platform", "indicator", "plspm_weight", "smartpls_weight", "abs_diff", "plspm_sig", "smartpls_sig"]]
    if len(failures) > 0:
        print("\n  Cells failing at least one criterion:", flush=True)
        print(failures.to_string(index=False), flush=True)


def main():
    parts = []
    for platform in PLATFORMS:
        df = run_platform(platform)
        parts.append(df)
        intermediate = os.path.join(LOG_DIR, f"stage0_plspm_partial_{platform}.csv")
        df.to_csv(intermediate, index=False)
        print(f"  intermediate -> {intermediate}", flush=True)
    full = pd.concat(parts, ignore_index=True)
    numeric_cols = ["plspm_weight", "plspm_boot_mean", "plspm_boot_se", "plspm_t_stat",
                    "plspm_ci_lo", "plspm_ci_hi", "plspm_p_value", "smartpls_weight", "abs_diff"]
    for c in numeric_cols:
        full[c] = pd.to_numeric(full[c], errors="coerce").round(4)
    full.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(full)} rows)", flush=True)
    assess(full)


if __name__ == "__main__":
    main()

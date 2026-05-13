"""Concatenate the four per-platform stage0 partial CSVs into the final
comparison CSV, then print and write the success-criteria assessment."""
import os
import pandas as pd
import numpy as np

from stage0_plspm import assess

REPO = "/Users/minsucho/Documents/Helpfulness/revisions"
LOG_DIR = os.path.join(REPO, "results", "robustness")
OUT_CSV = os.path.join(LOG_DIR, "stage0_plspm_vs_smartpls.csv")
PLATFORMS = ["amazon", "audible", "coursera", "hotel"]

def main():
    parts = []
    for p in PLATFORMS:
        path = os.path.join(LOG_DIR, f"stage0_plspm_partial_{p}.csv")
        parts.append(pd.read_csv(path))
    full = pd.concat(parts, ignore_index=True)
    numeric_cols = ["plspm_weight", "plspm_boot_mean", "plspm_boot_se", "plspm_t_stat",
                    "plspm_ci_lo", "plspm_ci_hi", "plspm_p_value", "smartpls_weight", "abs_diff"]
    for c in numeric_cols:
        if c in full.columns:
            full[c] = pd.to_numeric(full[c], errors="coerce").round(4)
    full.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}  ({len(full)} rows)")
    assess(full)


if __name__ == "__main__":
    main()

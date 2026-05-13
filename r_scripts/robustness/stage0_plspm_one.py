"""Run stage0_plspm.run_platform for a single platform passed on argv.

Writes a per-platform partial CSV to
results/robustness/stage0_plspm_partial_<platform>.csv

Used to overlap plspm fits with the still-running feature build:
    python3 r_scripts/robustness/stage0_plspm_one.py amazon
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage0_plspm import run_platform, LOG_DIR


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 stage0_plspm_one.py <platform>")
        sys.exit(1)
    platform = sys.argv[1]
    df = run_platform(platform)
    out = os.path.join(LOG_DIR, f"stage0_plspm_partial_{platform}.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()

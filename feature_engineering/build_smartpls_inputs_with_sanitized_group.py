

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sanitize_group_for_smartpls import (
    sanitize_csv, build_mapping_from_groups, write_mapping, load_mapping, UNSAFE
)

REPO = "/Users/minsucho/Documents/Helpfulness/revisions"
FEAT_DIR = os.path.join(REPO, "data", "robustness", "features_baseline")
SP_BASE  = os.path.join(REPO, "data", "robustness", "smartpls_input")
PLATFORMS = ["amazon", "audible", "coursera", "hotel"]

SYS_COLS_OUT = ["Depth", "Breadth", "Readability", "Arousal"]
HEU_COLS_OUT_FULL  = ["RatingDeviation", "TitleLength", "Recency"]
HEU_COLS_OUT_COURS = ["RatingDeviation", "Recency"]
RENAME = {
    "depth": "Depth",
    "breadth": "Breadth",
    "readability": "Readability",
    "arousal": "Arousal",
    "rating_deviation": "RatingDeviation",
    "title_length": "TitleLength",
    "recency": "Recency",
}


def canonical_mapping_path(platform):
    return os.path.join(SP_BASE, f"group_mapping_{platform}.csv")


def ensure_canonical_mapping(platform):
    """Build or load the canonical mapping for this platform. The mapping is
    derived from the baseline features CSV (which has the full universe of
    Group values for that platform)."""
    mapping_path = canonical_mapping_path(platform)
    if os.path.exists(mapping_path):
        mapping = load_mapping(mapping_path)
    else:
        feat = pd.read_csv(os.path.join(FEAT_DIR, f"{platform}.csv"))
        mapping = build_mapping_from_groups(feat["Group"])
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        write_mapping(mapping, mapping_path)
    return mapping_path


def build_baseline_smartpls_csv(platform, mapping_path):
    """Build the platform's baseline SmartPLS input CSV (all 7 indicators,
    K=10 KL breadth) at smartpls_input/baseline/<platform>_smartpls.csv."""
    feat = pd.read_csv(os.path.join(FEAT_DIR, f"{platform}.csv"))
    is_coursera = (platform == "coursera")
    heu_cols = HEU_COLS_OUT_COURS if is_coursera else HEU_COLS_OUT_FULL

    # Rename lowercase -> manuscript-style column names
    out = feat.rename(columns=RENAME).copy()
    cols = SYS_COLS_OUT + heu_cols + ["Helpfulness", "Group"]
    out = out[cols]

    raw_dir = os.path.join(SP_BASE, "baseline")
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, f"{platform}.csv")
    out.to_csv(raw_path, index=False)

    # Sanitize Group via the canonical mapping
    sanitized_path = os.path.join(raw_dir, f"{platform}_smartpls.csv")
    res = sanitize_csv(
        input_path=raw_path,
        output_path=sanitized_path,
        use_existing_mapping=mapping_path,
    )
    # Clean up the un-sanitized intermediate to avoid confusion
    os.remove(raw_path)
    return res


def sanitize_existing(platform, sub_dir):
    """Sanitize an existing pre-generated SmartPLS input CSV in place,
    writing <platform>_smartpls.csv alongside it. The pre-existing
    <platform>.csv (unsanitized) is left on disk for reference because the
    sub-spec generator built it from clean data."""
    src = os.path.join(SP_BASE, sub_dir, f"{platform}.csv")
    dst = os.path.join(SP_BASE, sub_dir, f"{platform}_smartpls.csv")
    return sanitize_csv(
        input_path=src,
        output_path=dst,
        use_existing_mapping=canonical_mapping_path(platform),
    )


def main():
    print("=== Step 1: build canonical per-platform mappings ===")
    mapping_paths = {}
    for platform in PLATFORMS:
        path = ensure_canonical_mapping(platform)
        mapping_paths[platform] = path
        m = pd.read_csv(path)
        print(f"  {platform:8s}: {len(m)} unique groups  -> {path}")

    results = []

    print("\n=== Step 2: build sanitized baseline SmartPLS CSVs ===")
    for platform in PLATFORMS:
        r = build_baseline_smartpls_csv(platform, mapping_paths[platform])
        results.append(("baseline", platform, r))
        print(f"  baseline/{platform}: rows={r['rows']}  unique_in={r['unique_groups_in']}  unique_out={r['unique_groups_out']}")

    print("\n=== Step 3: sanitize existing A2 + D1 CSVs ===")
    for sub_dir in ["A2-a", "A2-b", "A2-c", "D1"]:
        for platform in PLATFORMS:
            r = sanitize_existing(platform, sub_dir)
            results.append((sub_dir, platform, r))
            print(f"  {sub_dir:5s}/{platform}: rows={r['rows']}  unique_in={r['unique_groups_in']}  unique_out={r['unique_groups_out']}")

    # Final verification
    print("\n=== Verification across all 20 sanitized files ===")
    any_failed = False
    for sub_dir, platform, r in results:
        path = r["output"]
        df = pd.read_csv(path)
        n = len(df)
        nu = df["Group"].nunique()
        bad = df["Group"].astype(str).str.contains(UNSAFE).any()
        starts_ok = df["Group"].astype(str).str.match(r"^item_\d{4,}$").all()
        status = "OK"
        if bad or not starts_ok or n != r["rows"]:
            status = "FAIL"
            any_failed = True
        print(f"  {sub_dir:9s} {platform:8s}: rows={n}  unique={nu}  unsafe={bad}  all_item_ids={starts_ok}  {status}")
    if any_failed:
        sys.exit("\nVERIFICATION FAILED on at least one file. See above.")
    print("\nAll 20 SmartPLS input CSVs are sanitized and verified.")


if __name__ == "__main__":
    main()



import argparse
import os
import re
import sys
import pandas as pd
import numpy as np

ID_PATTERN = re.compile(r"^item_\d{4,}$")
UNSAFE = re.compile(r'[",\r\n]')


def build_mapping_from_groups(groups: "pd.Series") -> "pd.DataFrame":
    """Return a DataFrame with (item_id, original_group) for each unique value,
    in order of first appearance. Already-safe item_XXXX values map to themselves."""
    seen = []
    seen_set = set()
    for g in groups:
        if g not in seen_set:
            seen.append(g)
            seen_set.add(g)
    rows = []
    next_id = 1
    used_ids = set()
    # First pass: keep existing item_XXXX IDs as-is
    for g in seen:
        if isinstance(g, str) and ID_PATTERN.match(g):
            rows.append((g, g))
            used_ids.add(g)
    # Second pass: assign new IDs to the rest, skipping any IDs already used
    for g in seen:
        if isinstance(g, str) and ID_PATTERN.match(g):
            continue
        while True:
            candidate = f"item_{next_id:04d}"
            next_id += 1
            if candidate not in used_ids:
                used_ids.add(candidate)
                rows.append((candidate, g))
                break
    return pd.DataFrame(rows, columns=["item_id", "original_group"])


def load_mapping(mapping_path: str) -> "pd.DataFrame":
    df = pd.read_csv(mapping_path)
    if not set(["item_id", "original_group"]).issubset(df.columns):
        raise ValueError(f"Mapping file {mapping_path} must have columns 'item_id' and 'original_group'")
    return df[["item_id", "original_group"]]


def write_mapping(mapping: "pd.DataFrame", mapping_path: str) -> None:
    mapping.to_csv(mapping_path, index=False)


def sanitize_csv(input_path: str,
                 output_path: str = None,
                 mapping_path: str = None,
                 use_existing_mapping: str = None) -> dict:
    """Sanitize the Group column in `input_path` and write the sanitized CSV.

    Returns a small dict with row counts and verification flags.
    """
    df = pd.read_csv(input_path)
    if "Group" not in df.columns:
        raise ValueError(f"{input_path} has no 'Group' column")

    n_rows = len(df)
    unique_orig = df["Group"].nunique(dropna=False)

    # Build or load the mapping
    if use_existing_mapping:
        mapping = load_mapping(use_existing_mapping)
        # Verify coverage: every Group value in the input must be in the mapping
        missing = set(df["Group"].unique()) - set(mapping["original_group"])
        if missing:
            # Extend the mapping deterministically with new IDs that don't collide
            extra = build_mapping_from_groups(pd.Series(sorted(missing, key=str)))
            # Re-pick IDs to avoid collisions with the loaded mapping
            existing_ids = set(mapping["item_id"])
            new_id_start = 1
            new_rows = []
            for _, row in extra.iterrows():
                while True:
                    candidate = f"item_{new_id_start:04d}"
                    new_id_start += 1
                    if candidate not in existing_ids:
                        new_rows.append((candidate, row["original_group"]))
                        existing_ids.add(candidate)
                        break
            mapping = pd.concat([mapping, pd.DataFrame(new_rows, columns=["item_id", "original_group"])],
                                ignore_index=True)
            # Persist the extended mapping back to the canonical file
            write_mapping(mapping, use_existing_mapping)
    else:
        mapping = build_mapping_from_groups(df["Group"])

    lookup = dict(zip(mapping["original_group"], mapping["item_id"]))
    df["Group"] = df["Group"].map(lookup)

    if df["Group"].isna().any():
        raise ValueError(f"Some Group values could not be mapped in {input_path}")

    # Verification: ensure no row has a comma/quote/newline in Group
    bad = df["Group"].astype(str).str.contains(UNSAFE)
    if bad.any():
        raise ValueError(f"After sanitization, {bad.sum()} Group values in {input_path} still contain CSV-unsafe characters")

    if output_path is None:
        stem, ext = os.path.splitext(input_path)
        output_path = stem + "_smartpls" + ext
    df.to_csv(output_path, index=False, quoting=0)  # quoting=0 = csv.QUOTE_MINIMAL (default), no unnecessary quotes

    # Write mapping file (only if not reusing an existing mapping — else the
    # canonical mapping is the source of truth and is already on disk).
    if mapping_path is not None:
        write_mapping(mapping, mapping_path)
    elif use_existing_mapping is None:
        stem, ext = os.path.splitext(input_path)
        mapping_path = stem + "_group_mapping" + ext
        write_mapping(mapping, mapping_path)

    return {
        "input": input_path,
        "output": output_path,
        "mapping_path": mapping_path if mapping_path else use_existing_mapping,
        "rows": n_rows,
        "unique_groups_in": unique_orig,
        "unique_groups_out": int(df["Group"].nunique()),
        "any_unsafe_chars_after_sanitize": bool(bad.any()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("inputs", nargs="+", help="Input CSV path(s)")
    p.add_argument("--output", "-o", default=None,
                   help="Output path (single-input only). Default: <input>_smartpls.csv")
    p.add_argument("--mapping-out", default=None,
                   help="Where to write the mapping (single-input only). Default: <input>_group_mapping.csv")
    p.add_argument("--use-mapping", default=None,
                   help="Reuse this existing mapping CSV instead of building a new one. Extends it if it's missing any Group value.")
    args = p.parse_args()

    if len(args.inputs) > 1 and (args.output or args.mapping_out):
        sys.exit("--output and --mapping-out are only valid with a single input file")

    for inp in args.inputs:
        res = sanitize_csv(
            input_path=inp,
            output_path=args.output,
            mapping_path=args.mapping_out,
            use_existing_mapping=args.use_mapping,
        )
        print(f"  {res['input']}")
        print(f"    -> {res['output']}")
        if res['mapping_path']:
            print(f"    mapping: {res['mapping_path']}")
        print(f"    rows={res['rows']}  unique_groups: in={res['unique_groups_in']} out={res['unique_groups_out']}")
        print(f"    unsafe chars remaining: {res['any_unsafe_chars_after_sanitize']}")


if __name__ == "__main__":
    main()

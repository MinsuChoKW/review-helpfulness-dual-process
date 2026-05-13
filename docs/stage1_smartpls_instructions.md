## SmartPLS 4 Instructions for Stages 1 (A2) and 3 (D1)

After the Stage 0 verification confirmed that the R/Python `plspm` family
disagrees with SmartPLS 4 on weak formative indicators (see
[docs/stage0_verification_notes.md](stage0_verification_notes.md) §"R plspm
re-verification"), we invoke the contingency plan and run Stages 1 and 3
inside the SmartPLS 4 GUI. This document is the runbook.

**Total scope: 16 SmartPLS GUI runs.** Stages 2 and 4 are unaffected and run
separately in R (see `r_scripts/robustness/stage_c4.R` and `stage_d2.R`).

---

### 1. Input CSVs (already prepared)

All 20 SmartPLS-import CSVs live in `data/robustness/smartpls_input/`:

| Sub-spec | Path                                                              | Files (4 per sub-spec) |
|----------|-------------------------------------------------------------------|------------------------|
| baseline | `data/robustness/smartpls_input/baseline/<platform>_smartpls.csv` | amazon, audible, coursera, hotel |
| A2-a     | `data/robustness/smartpls_input/A2-a/<platform>_smartpls.csv`     | amazon, audible, coursera, hotel |
| A2-b     | `data/robustness/smartpls_input/A2-b/<platform>_smartpls.csv`     | amazon, audible, coursera, hotel |
| A2-c     | `data/robustness/smartpls_input/A2-c/<platform>_smartpls.csv`     | amazon, audible, coursera, hotel |
| D1       | `data/robustness/smartpls_input/D1/<platform>_smartpls.csv`       | amazon, audible, coursera, hotel |

The baseline set is included so that, if the SmartPLS project needs to be
rebuilt from scratch (e.g., to confirm the manuscript's Tables 5/6 latent
scores), the same sanitized inputs can be used.

Generators (re-runnable, idempotent):
- [feature_engineering/build_a2_breadth_variants.py](../feature_engineering/build_a2_breadth_variants.py) — builds the raw A2 + D1 CSVs
- [feature_engineering/sanitize_group_for_smartpls.py](../feature_engineering/sanitize_group_for_smartpls.py) — the reusable sanitization utility
- [feature_engineering/build_smartpls_inputs_with_sanitized_group.py](../feature_engineering/build_smartpls_inputs_with_sanitized_group.py) — builds canonical mappings, applies them to all 5 sub-specs, runs verification

#### Group column: sanitized to `item_XXXX`

SmartPLS 4's CSV reader does not honour CSV-standard escaping of embedded
commas inside quoted Group strings. Amazon's product names (e.g. *"Panasonic
Portable AM / FM Radio, Battery Operated Analog Radio, AC Powered, Silver
(RF-2400D) 22.8 x 7.8 x 10.8"*) caused SmartPLS to mis-split rows. To work
around this — and as a defensive measure for the other platforms — every
Group value across every sub-spec has been remapped to a deterministic,
4-digit-padded ID of the form `item_XXXX`.

The remapping is **canonical per platform**: the same `item_id` always
refers to the same original Group string regardless of sub-spec. The
canonical mappings are at:

```
data/robustness/smartpls_input/group_mapping_amazon.csv     (77 items)
data/robustness/smartpls_input/group_mapping_audible.csv    (100 items)
data/robustness/smartpls_input/group_mapping_coursera.csv   (205 items)
data/robustness/smartpls_input/group_mapping_hotel.csv      (33 items)
```

Each mapping file has two columns: `item_id, original_group`. Use it to
interpret platform-specific findings (e.g., to identify which
`item_0017` corresponds to which product) after the SmartPLS exports are
merged downstream. The downstream merge helper consumes the sanitized
SmartPLS exports as-is — the `item_XXXX` IDs are persistent across the
whole pipeline; only convert back to original strings when reporting
specific items by name.

#### What's inside each CSV

Column order (all sub-specs except where noted):

```
Depth, Breadth, Readability, Arousal, RatingDeviation, TitleLength, Recency, Helpfulness, Group
```

- For **coursera**, `TitleLength` is omitted in every sub-spec (no `Review_Title` in cleaned data).
- For **D1**, `Recency` is also omitted (so coursera D1 has only `RatingDeviation` as its Heuristic indicator — flagged below).

The non-`Breadth` indicator columns are byte-identical to those used in Stage 0
(reused from `data/robustness/features_baseline/<platform>.csv`). Only the
`Breadth` column changes between the four sub-specs:

| Sub-spec | Breadth definition                                                  |
|----------|---------------------------------------------------------------------|
| A2-a     | NMF K=5, base-10 KL of per-doc topic mixture against corpus mean   |
| A2-b     | NMF K=15, base-10 KL                                               |
| A2-c     | Topic entropy `-Σ p log10 p` on the baseline K=10 NMF mixtures      |
| D1       | Baseline K=10 KL (unchanged from Stage 0); Recency removed entirely |

All NMF runs use the same `random_state=42`, `init='nndsvd'`,
`TfidfVectorizer(stop_words="english", max_features=5000)`. The K=10 KL
recompute exactly matched the Stage 0 baseline `Breadth` to within 2.2 × 10⁻¹⁶
on every platform (sanity check in
[logs/build_a2_d1.log](../logs/build_a2_d1.log)).

`Helpfulness` is the raw integer count (outcome). `Group` is the item-name
string used as the multilevel grouping variable downstream. Neither is
included in the PLS model — they pass through unchanged so the downstream
`glmmTMB` step receives them.

---

### 2. SmartPLS 4 workflow (do this once per CSV — 16 runs total)

#### 2.1 Create / open a project

- Use a single SmartPLS project per **(sub-spec × platform)** combination.
  Name them: `A2-a_amazon`, `A2-a_audible`, …, `D1_hotel` (16 projects).
- Inside each project, import the matching CSV from
  `data/robustness/smartpls_input/<sub_spec>/<platform>_smartpls.csv`
  (note the `_smartpls` suffix — the file without it is either absent or
  un-sanitized and will fail to import on Amazon).
- Delimiter: `,`  ·  Value quote character: `"`  ·  Missing value marker: leave blank
  (the CSVs have no NAs in any indicator column — verified at generation time).

#### 2.2 Define the formative measurement model

Two latent constructs:

- **Systematic** (formative, Mode B): `Depth`, `Breadth`, `Readability`, `Arousal`
- **Heuristic**  (formative, Mode B): `RatingDeviation`, `TitleLength`, `Recency`

Modifications by sub-spec:

| Sub-spec | Systematic indicators (no change) | Heuristic indicators                                  |
|----------|-----------------------------------|--------------------------------------------------------|
| A2-a / A2-b / A2-c | Depth, Breadth, Readability, Arousal | RatingDeviation, TitleLength, Recency       |
| D1       | Depth, Breadth, Readability, Arousal | RatingDeviation, TitleLength (Recency dropped)        |

Per-platform exceptions (apply to all sub-specs):

- **coursera** drops `TitleLength` from Heuristic on every sub-spec (no title data).
- **coursera × D1** → Heuristic has a **single indicator** (`RatingDeviation` only).
  This is a valid but edge formative specification (single-indicator block is
  numerically equivalent to a standardized observed variable). Flag this as a
  caveat in the manuscript — the coursera D1 result is suggestive rather than
  decisive.

#### 2.3 Inner model (structural path)

Same as baseline: a placeholder outcome construct receives paths from both
Systematic and Heuristic. The outcome construct's measurement block contains a
single indicator (`Helpfulness`, Mode A). The paths and the outcome construct
exist only so that PLS-SEM has a valid inner model — they are not used
downstream; we only extract the latent scores for Systematic and Heuristic.

```
Systematic ──┐
             ▼
            Helpfulness  ←  single-indicator outcome (Mode A)
             ▲
Heuristic ───┘
```

#### 2.4 Run the PLS algorithm

- Algorithm: **PLS-SEM**
- Weighting scheme: **Path weighting** (this is SmartPLS 4 default).
- Maximum iterations: **300** (default).
- Stop criterion: **10⁻⁷** (default).
- Initial weights: **+1** (default).
- Standardize data: **Yes** (default; this matches what we did upstream for
  the open-source plspm runs).

#### 2.5 Run bootstrap

- Subsamples: **1000**
- Sign changes: **No sign changes** (this matches the convention SmartPLS used
  for the baseline; the Stage 0 verification showed that `plspm`'s
  majority-vote sign convention is what causes the breadth disagreement, so
  do *not* enable sign-change correction here).
- Confidence interval method: **Percentile bootstrap** (default).
- Test type: **Two-tailed**, **α = 0.05**.

#### 2.6 Export latent scores

After the PLS algorithm completes (the bootstrap is only needed for
significance reporting, not for the latent-score export):

- Right-click the Systematic LV → **Export scores** → save as
  `data/robustness/latent_<sub_spec>/<platform>_systematic.csv`
  (single column, one value per row, in the same row order as the input CSV).
- Right-click the Heuristic LV → **Export scores** → save as
  `data/robustness/latent_<sub_spec>/<platform>_heuristic.csv`.

Then run the merge helper (Python) — see §3 below — which produces the
unified `data/robustness/latent_<sub_spec>/<platform>.csv` file that the
downstream R scripts consume.

#### 2.7 Optional — record the outer weights for the Stage 1 reporting table

- **Model → Outer weights** (Path Coefficients tab) → copy the `Breadth`
  weight (and its bootstrap p-value, percentile CI bounds) into a row of
  `results/robustness/A2_breadth_alternative.csv`.
  Plan §"Reporting" wants the breadth weight in the systematic construct
  under each sub-spec, per platform.

---

### 3. Merging exported SmartPLS scores into the downstream input format

The downstream `glmmTMB` script expects a CSV with these columns:

```
Latent_Heuristic, Latent_Systematic, Group, Helpfulness
```

…in the same row order as the SmartPLS input CSV. The two single-column
exports from §2.6 plus the original `Group` and `Helpfulness` columns are
all we need.

After all 16 SmartPLS runs finish, run:

```bash
python3 feature_engineering/merge_smartpls_exports.py
```

(This script will be created in the next prep round; not strictly part of
this Stage-1 prep document. It will simply concatenate the per-LV exports
back into the layout that matches `data/latent_data/<platform>.csv`.)

The expected output locations are:

| Sub-spec | Output path                                                     |
|----------|------------------------------------------------------------------|
| A2-a     | `data/robustness/latent_A2-a/<platform>.csv`                    |
| A2-b     | `data/robustness/latent_A2-b/<platform>.csv`                    |
| A2-c     | `data/robustness/latent_A2-c/<platform>.csv`                    |
| D1       | `data/robustness/latent_D1/<platform>.csv`                      |

Each file: 4 columns (`Latent_Heuristic`, `Latent_Systematic`, `Group`,
`Helpfulness`) and one row per review, ordered identically to the input.

---

### 4. Sanity checks to run after each SmartPLS export (3 min per file)

Quick smoke tests before declaring the export valid:

1. **Row count match.** The exported scores file should have exactly the
   same row count as the input CSV
   (`wc -l data/robustness/smartpls_input/<sub>/<plat>.csv` minus the header).
2. **Latent score scale.** Each exported LV column should have mean ≈ 0 and
   sd ≈ 1 (SmartPLS standardizes by default).
3. **Sign of dominant indicator.** Open the outer-weights tab — `Depth`
   (Systematic) and `RatingDeviation` (Heuristic) should both have positive
   weights on every platform (these are the dominant indicators and their
   sign is stable across all engines).
4. **No NaNs / Infs.** `head` and `tail` the exported file — values should
   all be finite floats.

If any check fails, re-import the CSV from disk (don't reuse stale data) and
rerun the PLS algorithm.

---

### 5. Estimated SmartPLS effort

About 5 minutes per project once the workflow is muscle-memory, including
the import, model construction, algorithm run, bootstrap run, and the two
score exports. 16 projects × 5 min ≈ **80 minutes of GUI work**.

If the per-project cost is unacceptable, the plan's contingency-of-the-
contingency applies (`docs/robustness_plan.md` §"Contingency"): cut A2 to a
single sub-spec (A2-a only), reducing the SmartPLS load by 2/3. Recommend
trying the full 16-run path first.

---

### 6. Where to report when SmartPLS is done

- After step 3 produces all `data/robustness/latent_<sub_spec>/<platform>.csv`
  files, signal Claude to run Stage 1 (A2) and Stage 3 (D1) Multilevel ZINB
  via `r_scripts/robustness/stage_a2_zinb.R` and `stage_d1_zinb.R`
  (these will be authored in the next round, paralleling
  `r_scripts/multilevel_zinb_modeling.R`).
- The final outputs of those R scripts are:
  - `results/robustness/A2_breadth_alternative.csv`
  - `results/robustness/D1_recency_removed.csv`

Stages 2 (C4) and 4 (D2) are already running in parallel in R — see
[results/robustness/C4_exposed_subsample.csv](../results/robustness/C4_exposed_subsample.csv)
and [results/robustness/D2_item_FE.csv](../results/robustness/D2_item_FE.csv)
once those complete.

# Information Processing and Review Helpfulness in Online Platforms: A Dual-Process Perspective

Replication code and derived data for the IEEE Access manuscript of the same
title. The study models online review helpfulness as a two-stage cognitive
process — heuristic gatekeeping (review visibility) followed by systematic
evaluation (helpfulness rating) — and validates the framework on roughly
400,000 reviews from four platforms (Amazon, Booking.com, Audible, Coursera)
using formative PLS-SEM (SmartPLS 4) and Multilevel Zero-Inflated Negative
Binomial regression (R, `glmmTMB`).

## Citation

> Cho, M. et al. *Information Processing and Review Helpfulness in Online
> Platforms: A Dual-Process Perspective.* IEEE Access, forthcoming.
> (DOI placeholder — will be populated once the article is assigned.)

A BibTeX entry will be added to this README when the DOI is issued.

## Overview

Heuristic cues (rating deviation, title length, recency) and systematic cues
(content depth, content breadth, readability, emotional arousal) are
extracted from each review's text and metadata. A formative PLS-SEM model in
SmartPLS 4 combines those seven indicators into two latent constructs,
`Latent_Heuristic` and `Latent_Systematic`. A Multilevel ZINB model in R
then estimates the effect of each construct on review helpfulness, with the
zero-inflation component capturing the visibility (gatekeeping) stage and
the conditional count component capturing the evaluation stage. The
multilevel structure uses the reviewed item as the grouping variable.

Robustness is checked in four families (manuscript Section VI.D):

1. **A2** — three alternative breadth operationalisations (NMF K=5,
   NMF K=15, topic entropy on K=10).
2. **C4** — exposure-conditional sub-sample analysis on the rows with
   `Helpfulness > 0`.
3. **D1** — heuristic block re-estimated without the recency indicator.
4. **D2** — within-item-centered ZINB to remove between-item variance.

## Data availability

| Source | What's in this repo | What's not, and why |
|--------|---------------------|---------------------|
| Amazon Reviews'23 | Cleaned per-review CSV with raw review text (`data/cleaned_data/amazon.csv`), derived per-review indicator features, and latent construct scores. | The full Amazon Reviews'23 corpus is openly redistributable from McAuley et al.; we ship only the subset used here. |
| Booking.com hotels | Derived per-review indicator features and latent construct scores (`data/latent_data/hotel.csv`, `data/robustness/features_baseline/hotel.csv`, `data/robustness/smartpls_input/*/hotel*_smartpls.csv`). | Raw review text is restricted by Booking.com's Terms of Service and is not redistributed. |
| Audible audiobook reviews | Derived per-review indicator features and latent scores (`audible.csv` analogues). | Raw review text is restricted by Audible's Terms of Service. |
| Coursera course reviews | Derived per-review indicator features and latent scores (`coursera.csv` analogues). | Raw review text is restricted by Coursera's Terms of Service. |

The `Group` column for the three restricted platforms in `data/latent_data/`
and `data/robustness/features_baseline/` carries the item identifier (hotel
name, book title, course slug). These are publicly visible on the
respective platforms; they are kept here so the manuscript's openly-available
latent score files can be matched to the platform-side records. The
SmartPLS-input CSVs (`data/robustness/smartpls_input/`) and SmartPLS exports
(`smartpls/results/`) use a sanitised `item_XXXX` identifier instead; see
the canonical mapping files `data/robustness/smartpls_input/group_mapping_<platform>.csv`
to translate between the two.

## Repository structure

```
.
├── README.md                       this file
├── LICENSE                         MIT (covers code and documentation only)
├── .gitignore
├── requirements.txt                Python dependencies
├── data/
│   ├── cleaned_data/               raw review-level CSV (only Amazon is public)
│   ├── latent_data/                4 platform CSVs with the baseline PLS-SEM latent scores
│   └── robustness/
│       ├── features_baseline/      7 baseline indicator features per review, per platform
│       └── smartpls_input/         sanitised SmartPLS-ready inputs (one per A2 sub-spec, D1, baseline)
│           ├── group_mapping_<platform>.csv   canonical item_XXXX ↔ original-Group mapping
│           ├── baseline/, A2-a/, A2-b/, A2-c/, D1/
├── feature_engineering/
│   ├── make_features.py                            baseline 7-indicator feature pipeline
│   ├── build_features_for_stage0.py                regenerates baseline features for Stage 0 verification
│   ├── build_a2_breadth_variants.py                builds the three A2 breadth variants + D1 input CSVs
│   ├── build_smartpls_inputs_with_sanitized_group.py  applies canonical sanitisation across all 20 SmartPLS inputs
│   ├── sanitize_group_for_smartpls.py              standalone Group-column sanitiser (utility)
│   └── emotion_va_scores.csv                       go_emotions label → arousal score table for the Arousal indicator
├── r_scripts/
│   ├── zinb_modeling.R             baseline Standard ZINB (manuscript Table 5)
│   ├── multilevel_zinb_modeling.R  baseline Multilevel ZINB (manuscript Table 6)
│   ├── baseline_zinb_coefs.R       captures Tables 5 & 6 coefficients for Figure 3
│   └── robustness/
│       ├── run_all_zinb.R          canonical driver for all 24 robustness fits (A2 × 12, D1 × 4, C4 × 4, D2 × 4)
│       ├── stage0_plspm.py         Stage 0 verification: Python plspm port baseline reproduction
│       ├── stage0_plspm_R.R        Stage 0 verification: R plspm baseline reproduction
│       └── stage_c4.R, stage_d2.R  per-stage stand-alone R drivers (subsumed by run_all_zinb.R)
├── smartpls/
│   └── results/
│       ├── latent_score/{a2-a,a2-b,a2-c,d1}/  exported Latent_Heuristic & Latent_Systematic per review
│       ├── outer_weight/{a2-a,a2-b,a2-c,d1}/  exported indicator weights with bootstrap p-values
│       └── raw_file/                           sanitised SmartPLS-ready inputs used to produce the exports
├── results/
│   ├── baseline_zinb_coefficients.csv          Tables 5 + 6 coefficients (32 rows)
│   └── robustness/
│       ├── all_specs_coefficients.csv          88-row long-format table of every robustness coefficient
│       ├── summary_table.csv                   per-spec sign-/sig-concordance vs baseline
│       ├── A2_breadth_weights.csv              breadth outer-weight comparison vs baseline (Table 2)
│       ├── D1_no_recency_heuristic.csv         D1 heuristic coefficient comparison vs baseline (Table 6)
│       ├── convergence_log.csv                 random-slopes → random-intercept fallbacks per spec
│       └── stage0_plspm_vs_smartpls.csv        Stage 0 engine-comparison verification
├── docs/
│   ├── robustness_plan.md              full robustness execution plan (v2)
│   ├── baseline_specification.md       answers to the Stage 0 "open questions"
│   ├── stage0_verification_notes.md    SmartPLS-vs-plspm divergence diagnosis
│   └── stage1_smartpls_instructions.md  SmartPLS GUI runbook for the 16 robustness runs
└── figures/
    ├── figure3_zinb_comparison.py      Standard vs Multilevel ZINB coefficient comparison
    └── figure3_zinb_comparison.pdf
```

## Software requirements

- **Python 3.9+** with the packages listed in [`requirements.txt`](requirements.txt).
  RoBERTa inference for the Arousal indicator uses `transformers` + `torch`;
  it runs on CPU but is materially faster with Apple Silicon MPS or CUDA.
- **R 4.2+** with packages `glmmTMB`, `dplyr`, `plspm`, `MuMIn`.
  Install with:
  ```r
  install.packages(c("glmmTMB", "dplyr", "plspm", "MuMIn"),
                   repos = "https://cran.r-project.org")
  ```
  Or via micromamba / conda, e.g.: `micromamba install -c conda-forge r-base r-glmmtmb r-dplyr r-plspm r-mumin`.
- **SmartPLS 4** (commercial, GUI-only) for the PLS-SEM estimation. SmartPLS
  is not scriptable; the latent score CSVs and outer-weight tables produced
  by the GUI are shipped here (in `smartpls/results/`) so the SmartPLS step
  does not need to be re-executed to reproduce the manuscript's tables.
  See [`docs/stage1_smartpls_instructions.md`](docs/stage1_smartpls_instructions.md)
  for the exact GUI workflow used.

## Reproducing the main analyses

### 1. Feature engineering (Python)

The seven per-review indicators (Depth, Breadth, Readability, Arousal,
RatingDeviation, TitleLength, Recency) are computed from the cleaned
per-platform CSVs:

```bash
# from the repository root
python3 feature_engineering/make_features.py
```

For platforms whose raw text is restricted (Booking.com, Audible, Coursera),
the cleaned CSVs cannot be regenerated locally; the resulting indicator
features are provided directly in `data/robustness/features_baseline/<platform>.csv`.

### 2. PLS-SEM in SmartPLS 4 (manual)

SmartPLS 4 is a GUI tool. The SmartPLS workflow is documented in
[`docs/stage1_smartpls_instructions.md`](docs/stage1_smartpls_instructions.md).
For pure-reproducibility purposes the SmartPLS-exported latent scores and
outer weights are committed under `smartpls/results/`, so the SmartPLS step
can be skipped.

### 3. ZINB modeling (R)

Reproduce manuscript Tables 5 and 6:

```r
# from the r_scripts/ directory
Rscript zinb_modeling.R              # Table 5 — Standard ZINB
Rscript multilevel_zinb_modeling.R   # Table 6 — Multilevel ZINB
```

To capture the coefficients into a single CSV (`results/baseline_zinb_coefficients.csv`)
that the figure script consumes:

```bash
Rscript r_scripts/baseline_zinb_coefs.R
```

## Reproducing the robustness analyses (Section VI.D)

A single R script fits all 24 robustness specifications (12 A2 sub-specs ×
platform, 4 D1, 4 C4, 4 D2):

```bash
Rscript r_scripts/robustness/run_all_zinb.R
```

Inputs:

- For the A2 sub-specs and D1, the SmartPLS-exported latent scores under
  `smartpls/results/latent_score/<sub-spec>/<sub-spec>_<platform>.csv`, with
  the `Group` column re-attached by row order from
  `data/robustness/smartpls_input/<SUB-SPEC>/<sub-spec>-<platform>_smartpls.csv`.
- For C4 and D2, the baseline latent scores at `data/latent_data/<platform>.csv`.

Outputs (written under `results/robustness/`):

- `all_specs_coefficients.csv` — 88 rows: 4 core coefficients per (spec, platform)
  combination for A2, D1, D2, plus 2 conditional coefficients for C4 (NB2 has no
  zero-inflation component).
- `summary_table.csv` — per-spec sign-/significance-concordance vs the
  manuscript Table 6 baseline.
- `A2_breadth_weights.csv` — breadth outer-weight comparison for the three
  A2 sub-specs vs manuscript Table 2.
- `D1_no_recency_heuristic.csv` — D1 heuristic conditional/ZI coefficient
  comparison vs Table 6, with the coursera single-indicator caveat flagged.
- `convergence_log.csv` — record of every spec that needed a random-slopes
  → random-intercept fallback.

The script falls back to a random-intercept-only model on any spec whose
random-slopes ZINB has a non-PD Hessian (about 80% of the 24 fits, including
all four C4 fits except one). Every reported model has a PD Hessian after
fallback.

Mapping the four robustness families to manuscript Section VI.D sub-sections:

| Manuscript | Repo spec key |
|------------|---------------|
| Section VI.D.1 — Alternative breadth operationalisations | `a2-a` (K=5), `a2-b` (K=15), `a2-c` (entropy) |
| Section VI.D.2 — Voted-review sub-sample analysis | `c4` |
| Section VI.D.3 — Recency-excluded heuristic | `d1` |
| Section VI.D.4 — Within-item identification | `d2` |

## Reproducing Figure 3

After running `Rscript r_scripts/baseline_zinb_coefs.R` (which produces
`results/baseline_zinb_coefficients.csv`):

```bash
python3 figures/figure3_zinb_comparison.py
```

This renders [`figures/figure3_zinb_comparison.pdf`](figures/figure3_zinb_comparison.pdf)
and a 600-dpi PNG copy.

## License

Source code and documentation files in this repository are licensed under the
[MIT License](LICENSE). The license does **not** cover the external review
datasets referenced or partially included here: Amazon Reviews'23 is openly
redistributable under McAuley et al.'s terms, while raw review text from
Booking.com, Audible, and Coursera is restricted by those platforms' Terms
of Service and is not included.

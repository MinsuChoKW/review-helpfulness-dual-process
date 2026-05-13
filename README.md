# Information Processing and Review Helpfulness in Online Platforms: A Dual-Process Perspective

Replication code and derived data for the IEEE Access manuscript of the same
title. The study models online review helpfulness as a two-stage cognitive
process — heuristic gatekeeping (review visibility) followed by systematic
evaluation (helpfulness rating) — and validates the framework on roughly
400,000 reviews from four platforms (Amazon, Booking.com, Audible, Coursera)
using formative PLS-SEM (SmartPLS 4) and Multilevel Zero-Inflated Negative
Binomial regression (R, `glmmTMB`).

## Overview

The repository accompanies the manuscript "Information Processing and Review
Helpfulness in Online Platforms: A Dual-Process Perspective", forthcoming in
IEEE Access. The study applies dual-process theory to review helpfulness
across four online platforms (Amazon, Booking.com, Audible, Coursera),
using formative PLS-SEM for latent construct estimation and Multilevel
ZINB models for outcome modeling. Robustness analyses across four families
of alternative specifications are reported in manuscript Section VI.D.

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

## Quick start

The pipeline runs in three stages:

1. **Feature engineering (Python)** — extract per-review indicators
   (depth, breadth, readability, arousal, rating deviation, title length,
   recency) from cleaned review data. Scripts: `feature_engineering/`.
2. **PLS-SEM (SmartPLS 4)** — estimate latent heuristic and systematic
   constructs from indicators. Project files: `smartpls/Helpfulness/`.
   Exported latent scores: `smartpls/results/latent_score/`.
3. **ZINB modeling (R)** — fit baseline and robustness models on the
   latent scores. Scripts: `r_scripts/`.

Each script has a header docstring stating its inputs, outputs, and
corresponding manuscript section.

## License

Source code and documentation files in this repository are licensed under the
[MIT License](LICENSE). The license does **not** cover the external review
datasets referenced or partially included here: Amazon Reviews'23 is openly
redistributable under McAuley et al.'s terms, while raw review text from
Booking.com, Audible, and Coursera is restricted by those platforms' Terms
of Service and is not included.

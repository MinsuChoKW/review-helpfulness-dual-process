## Baseline Specification — Answers to the Open Questions

These three questions in `docs/robustness_plan.md` (lines 407–420) must be resolved
before Stage 0 begins. Answers are derived from the existing code and data.

### Q1. NMF K used in the baseline breadth feature

**K = 10.**

Evidence: [feature_engineering/make_features.py:30](../feature_engineering/make_features.py#L30)
defines `compute_breadth_nmf(df, text_col, n_topics=10)` and the caller at
[feature_engineering/make_features.py:73](../feature_engineering/make_features.py#L73)
invokes it with no override, so the default `n_topics=10` is what was applied to all
platforms. The NMF call also uses `random_state=42, init='nndsvd'`, and breadth is
computed as a base-10 KL divergence of each review's NMF mixture against the corpus
mean mixture.

**Implication for Stage 1 (A2).** A2-a uses K=5, A2-b uses K=15, A2-c replaces the
KL aggregation with topic entropy `-Σ p_ij log p_ij` while leaving K at the baseline
value of 10. All other NMF hyperparameters (`random_state=42`, `init='nndsvd'`,
`max_features=5000`, English stopwords) should be held fixed across A2 sub-specs.

### Q2. Scaling/transformation in `data/latent_data/*.csv`

**The three latent score columns are standardized (mean ≈ 0, SD = 1.0). `Helpfulness`
and `Group` are kept on their original scales (raw count and item-name string,
respectively).** No further transformation is applied.

Evidence (column-level descriptive statistics on each file):

| Platform | Column              | mean    | sd    | min    | max    |
|----------|---------------------|---------|-------|--------|--------|
| amazon   | Helpfulness         |  0.222  | 3.148 |  0     | 589    |
|          | Latent_Helpfulness  |  0.000  | 1.000 | -0.070 | 187.05 |
|          | Latent_Heuristic    |  0.000  | 1.000 | -1.513 |   6.54 |
|          | Latent_Systematic   |  0.000  | 1.000 | -1.595 |  28.34 |
| audible  | Helpfulness         |  0.454  | 7.501 |  0     | 656    |
|          | Latent_Helpfulness  |  0.000  | 1.000 | -0.061 |  87.40 |
|          | Latent_Heuristic    |  0.000  | 1.000 | -1.947 |   5.28 |
|          | Latent_Systematic   |  0.000  | 1.000 | -1.434 |  32.62 |
| coursera | Helpfulness         |  0.174  | 2.106 |  0     | 239    |
|          | Latent_Helpfulness  |  0.000  | 1.000 | -0.083 | 113.38 |
|          | Latent_Heuristic    |  0.000  | 1.000 | -1.202 |   7.31 |
|          | Latent_Systematic   |  0.000  | 1.000 | -4.330 |  37.05 |
| hotel    | Helpfulness         |  0.108  | 0.377 |  0     |  14    |
|          | Latent_Helpfulness  |  0.000  | 1.000 | -0.286 |  36.83 |
|          | Latent_Heuristic    |  0.000  | 1.000 | -1.526 |   6.21 |
|          | Latent_Systematic   |  0.000  | 1.000 | -1.613 |  17.80 |

The heavy positive skew of the `Latent_*` maxima (e.g., 187 SDs for
`Latent_Helpfulness` on amazon) is a consequence of standardizing a small
formative composite that has a fat right tail in the original indicator scale —
not of any extra transformation.

**Implication for Stage 0.** `plspm` defaults to `scaled = TRUE` (column z-scoring
prior to PLS estimation), which reproduces what SmartPLS did. Therefore the seven
observed indicators should be passed to `plspm` on their raw computed scale and
`scaled` left at its default. Comparable behaviour holds in the Python `plspm` port
(`Config.add_lv(..., Mode.B, ...)` with default scaling). The latent scores
produced by `plspm` should also come out with mean 0 and SD 1, matching the latent
columns in `data/latent_data/*.csv`.

**Implication for Stages 1 and 3.** When regenerating `Latent_Heuristic` and
`Latent_Systematic`, do **not** apply any extra centering/scaling beyond `plspm`'s
defaults. The downstream `glmmTMB` baseline does not centre or scale the latent
columns, so any re-estimated latent column must arrive on the same z-scale to
remain coefficient-comparable to the manuscript's Tables 5 and 6.

### Q3. Recommended placeholder outcome for the `plspm` inner model

**Recommendation: option (a) — use `Helpfulness` itself as a single-indicator
endogenous construct.**

Reasons:
- `plspm` requires the inner-model matrix to define at least one path. The minimal
  valid structure is `Heuristic → Outcome` and `Systematic → Outcome`, with
  `Outcome` reflectively measured by a single observed indicator. Setting the
  measurement mode of that outcome block to `"A"` (reflective) with one indicator
  gives that indicator a loading of 1 and a weight of 1, so the outcome block adds
  no degrees of freedom that compete with the formative weights we care about.
- Using raw `Helpfulness` rather than `log1p(Helpfulness)` keeps the inner
  regression linear in the same outcome the downstream ZINB consumes, which makes
  the optional sanity check (do `plspm` inner-model path coefficients have the
  same signs as `glmmTMB`'s conditional fixed effects?) interpretable. The path
  coefficients themselves are not used in Stage 0 — only the outer (indicator)
  weights are evaluated against the manuscript's Tables 2 and 3 — but keeping the
  outcome on its original scale loses nothing.
- Option (b) `log1p(Helpfulness)` would be preferable only if the inner-model
  fit were itself the object of inference, which it is not at Stage 0.
- The choice has no effect on the formative outer weights of `Heuristic` and
  `Systematic` to within numerical noise, because in PLS-SEM the outer weights of
  a formative block are determined by the block's own indicator covariance
  structure and its inner-weighted construct score, and the latter is invariant
  to monotone rescaling of a one-indicator outcome.

**Operational consequence for Stage 0.** The `plspm` model is:

- Blocks: `Heuristic` (formative, mode B), `Systematic` (formative, mode B),
  `Outcome` (reflective, mode A, single indicator = `Helpfulness`).
- Inner path matrix: `Heuristic → Outcome`, `Systematic → Outcome`.
- Heuristic indicators: `rating_deviation`, `title_length`, `recency` (Coursera
  drops `title_length` per Plan §III.A.2).
- Systematic indicators: `depth`, `breadth`, `readability`, `arousal`.
- Inner weighting scheme: `"path"` (default in SmartPLS 4; not the `plspm`
  default which is centroid — must be set explicitly).
- Bootstrap: 1000 resamples for p-values of indicator weights.

This single configuration is reused unchanged for Stages 1 and 3, with only the
indicator lists rebuilt according to each sub-spec.

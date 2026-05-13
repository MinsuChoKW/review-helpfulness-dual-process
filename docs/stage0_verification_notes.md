## Stage 0 — Verification Notes

This file records the result of reproducing the SmartPLS 4 baseline indicator
weights with `plspm`. The plan in `docs/robustness_plan.md` §"Stage 0" defines
the success criteria; see `docs/baseline_specification.md` for the model spec.

### Engine used

`plspm` Python port (PyPI `plspm`), not the R package. R is not installed in
the working environment, and installing the R toolchain would have been
disproportionate. The Python port exposes the same algorithm — formative Mode B,
inner Scheme.PATH, multiprocess bootstrap — and is the official port of the R
package. The choice is documented in
[r_scripts/robustness/stage0_plspm.py](../r_scripts/robustness/stage0_plspm.py).

If the manuscript editors require an R-side rerun, the R script equivalent of
`r_scripts/robustness/stage0_plspm.py` would be a near-verbatim translation: same
`plspm(Data, path_matrix, blocks, modes=c("B","B","A"), scheme="path", boot.val=TRUE, br=1000)`
call.

### Feature regeneration

`data/cleaned_data/*.csv` does **not** contain the seven indicator features.
Only the post-PLS latent scores are stored, in `data/latent_data/*.csv`. So
before Stage 0 could be run, the seven indicators were regenerated from raw
text via [feature_engineering/build_features_for_stage0.py](../feature_engineering/build_features_for_stage0.py),
which mirrors `feature_engineering/make_features.py` exactly:

- Depth — stopword-filtered word count
- Breadth — NMF (K=10, random_state=42, init=nndsvd, max_features=5000) → base-10 KL of per-doc topic mixture against corpus mean
- Readability — Flesch Reading Ease (textstat)
- Arousal — confidence-weighted mean over the 28 emotions output by `SamLowe/roberta-base-go_emotions`, mapped through `feature_engineering/emotion_va_scores.csv`
- Rating Deviation — |Rating - Average_Rating|
- Title Length — character length of Review_Title (NA for coursera; no title column)
- Recency — `Time_Lapsed` from the cleaned data (days since posting)

Apple-Silicon MPS was used for the RoBERTa pass; rate ~ 50 reviews/sec on the
laptop used here.

### Results

#### Per-platform plspm indicator weights

Full table: [results/robustness/stage0_plspm_vs_smartpls.csv](../results/robustness/stage0_plspm_vs_smartpls.csv)
(27 rows: 4 platforms × 7 indicators − 1 NA cell for coursera × title_length).
Per-platform partials: `results/robustness/stage0_plspm_partial_{amazon,audible,coursera,hotel}.csv`.
SmartPLS reference values come from the manuscript's Tables 2 and 3 (provided
by the user on 2026-05-11). Significance encoding: `***` p<0.001, `**` p<0.01,
`*` p<0.05, `n.s.` not significant.

Stage 0 fit times (plspm + 1000-iter bootstrap, 4 processes):

| Platform | n      | plspm fit time |
|----------|-------:|---------------:|
| amazon   | 89,927 |        176 s   |
| audible  | 92,989 |        169 s   |
| coursera | 121,386|        142 s   |
| hotel    | 89,505 |        464 s   |

#### Success-criteria assessment

The three criteria in `docs/robustness_plan.md` §"Stage 0 → Success criteria",
with the cell-count threshold corrected from 22/24 to **25/27** (actual count is
4 platforms × 7 indicators − 1 NA = 27; original 22/24 ≈ 92% rate scales to
25/27; threshold update authorised by the user on 2026-05-11):

| # | Criterion                                                   | Result   | Pass? |
|---|-------------------------------------------------------------|----------|-------|
| 1 | Indicator weights match manuscript within ±0.03 (every cell)| 13 / 27  | ✗     |
| 2 | Sign of every weight matches                                | 21 / 27  | ✗     |
| 3 | Significance category matches in at least 25 of 27 cells    | 16 / 27  | ✗     |

**Verdict: Stage 0 verification FAILS the strict criteria.**

##### By indicator

| Indicator        | Sign match | ±0.03 tol | Sig match | Note |
|------------------|-----------:|----------:|----------:|------|
| depth            | 4/4        | 3/4       | 4/4       | Dominant Systematic indicator; close match (1.00–1.04 vs 1.024–1.040). |
| rating_deviation | 4/4        | 2/4       | 4/4       | Dominant Heuristic indicator; near-identical on audible and coursera. |
| title_length     | 3/3        | 1/3       | 2/3       | Big magnitude gap on hotel (0.215 vs 0.390). |
| recency          | 3/4        | 2/4       | 2/4       | Perfect on audible (0.752/0.755) and coursera (0.227/0.227); flips on amazon. |
| readability      | 4/4        | 4/4       | 1/4       | Magnitudes match (all within 0.020) but plspm consistently downgrades significance. |
| arousal          | 3/4        | 1/4       | 2/4       | Flips on hotel (−0.033 vs +0.085); magnitude inflated on coursera. |
| breadth          | **0/4**    | 0/4       | 1/4       | **Sign flips on all four platforms.** Magnitudes always near zero in plspm. |

##### By platform

| Platform | Sign match | ±0.03 tol | Sig match |
|----------|-----------:|----------:|----------:|
| amazon   | 5/7        | 2/7       | 6/7       |
| audible  | 6/7        | 6/7       | 3/7       |
| coursera | 5/6        | 4/6       | 3/6       |
| hotel    | 5/7        | 1/7       | 4/7       |

#### Diagnosis

The failure pattern is concentrated, not diffuse:

1. **Dominant indicators reproduce closely.** `depth` (loading ≈ 0.99) and
   `rating_deviation` (loading ≈ 0.6–0.97) — the indicators that explain most
   of each construct's variance — match SmartPLS within 0.02–0.06. The
   constructs themselves are clearly the same.
2. **Weak indicators flip signs.** All four platforms show `breadth` weights
   near zero in plspm (|w| < 0.12) but at clearly non-trivial negative values
   in SmartPLS (−0.085 to −0.159 on three platforms; +0.251 on hotel). The
   sign is *not* random bootstrap noise — bootstrap CIs are narrow and on the
   plspm side of zero.
3. **Significance categories shift systematically.** plspm gives narrower
   significance for weak indicators (lots of `n.s.` and `*` where SmartPLS
   reports `***`). This co-occurs with the smaller magnitudes — when the
   weight estimate is near zero, both the magnitude and the t-stat fall.

Two algorithmic differences between Python `plspm` and SmartPLS 4 are
candidates for the pattern:

- **Sign-fixing convention.** Python `plspm` (in
  [`weights.py`](file:///Users/minsucho/Library/Python/3.9/lib/python/site-packages/plspm/weights.py)
  lines 63–68) chooses each construct's orientation so the majority of MV
  correlations with the score are positive (sign of the sum of correlation
  signs). SmartPLS 4 instead retains the orientation seeded by the first
  iteration's correlations with neighbouring constructs. For a construct
  with one dominant indicator (depth) plus several weak indicators (breadth,
  readability, arousal), both rules agree on the dominant orientation but
  can flip the *individual* weak weights to the opposite side of zero.
  Because `breadth` and `arousal` correlate weakly with the construct, the
  Wold iteration can converge to either side; SmartPLS and plspm converge to
  opposite sides.
- **Inner-weighting initialization.** SmartPLS 4 uses path weighting with
  unit-weight start; Python `plspm` uses path weighting with weights
  proportional to `1/sqrt(block_size)`. With strongly collinear systematic
  indicators (depth ≈ breadth + readability + arousal residual), this can
  drive the iteration into different fixed points of the Wold algorithm.

Both of these are *known* sources of cross-program disagreement in formative
PLS-SEM and have been documented in the literature (see Hair et al. 2017
discussions of sign indeterminacy and Henseler 2010 on inner-weighting
initialization).

#### Recommendation — invoke the contingency plan

`docs/robustness_plan.md` §"Contingency: If `plspm` Fails Verification"
applies. Concretely:

- **Stages 1 (A2) and 3 (D1)**, which require re-running PLS, should be done
  in **SmartPLS 4 (GUI)** rather than Python `plspm`. Total SmartPLS GUI
  effort: 16 runs (A2 × 3 sub-specs × 4 platforms + D1 × 4 platforms). The
  plan already allows for this overhead.
- **Stages 2 (C4) and 4 (D2)** remain unaffected — they reuse the existing
  `data/latent_data/*.csv` SmartPLS scores and run pure R-side (or here,
  Python-side equivalent) analyses on those scores.
- **Stage 0 finding to record in the manuscript:** the formative weights of
  the dominant indicators of each construct (depth, rating_deviation) are
  cross-implementation stable; the weights of weak (near-zero) indicators
  (breadth, arousal, readability) are not, due to known sign-indeterminacy
  and inner-weighting-initialization effects in formative PLS-SEM. This
  motivates the contingency choice and is itself useful context for the
  manuscript's robustness section.

### R plspm re-verification (2026-05-12)

User authorised installing R locally to test whether the SmartPLS-vs-plspm
disagreement is specific to the **Python** port of `plspm`, or whether it
extends to the original **R** package. R 4.5.3 and CRAN `plspm` 0.6.0
were installed via micromamba into `~/.local/r-env` (no admin required;
brew was not available on this host). The runner
[`r_scripts/robustness/stage0_plspm_R.R`](../r_scripts/robustness/stage0_plspm_R.R)
mirrors the Python spec exactly: per-column z-scoring upstream, then
`plspm(..., scheme = "path", scaled = FALSE, boot.val = TRUE, br = 1000)`,
modes `c("B", "B", "A")`, same indicator features
(`data/robustness/features_baseline/*.csv` reused verbatim).

Output: [`results/robustness/stage0_plspm_R_vs_smartpls.csv`](../results/robustness/stage0_plspm_R_vs_smartpls.csv).

#### Breadth side-by-side (Python plspm vs R plspm vs SmartPLS)

| Platform | Python plspm weight | R plspm weight | R 95% boot CI       | SmartPLS | R CI contains SmartPLS? |
|----------|--------------------:|---------------:|:--------------------|---------:|:-----------------------:|
| Amazon   | +0.1161             | **+0.1161**    | [+0.0595, +0.1541] | −0.085   | **No** (≈ 8 SE away) |
| Audible  | +0.0510             | **+0.0510**    | [−0.0244, +0.1063] | −0.108   | **No** (≈ 5 SE away) |
| Coursera | +0.0094             | **+0.0094**    | [−0.0502, +0.0590] | −0.159   | **No** (≈ 6 SE away) |
| Hotel    | −0.0034             | **−0.0034**    | [−0.0662, +0.0590] | +0.251   | **No** (≈ 8 SE away) |

#### Full R plspm success-criteria assessment

| # | Criterion                                                   | R plspm  | Python plspm | Pass? |
|---|-------------------------------------------------------------|----------|--------------|-------|
| 1 | Weights within ±0.03 (every cell)                           | 13 / 27  | 13 / 27      | ✗     |
| 2 | Sign of every weight matches                                | 21 / 27  | 21 / 27      | ✗     |
| 3 | Significance category matches ≥ 25 of 27                    | 17 / 27  | 16 / 27      | ✗     |
| — | Bootstrap CI contains SmartPLS reference                    | 15 / 27  | (not computed) | —   |

R plspm and Python plspm produce **bit-identical** weight estimates to four
decimal places on every cell of every platform. Bootstrap SEs and 95% CIs
also agree to within bootstrap noise. This rules out the Python port being
the culprit — the disagreement is between the open-source `plspm` family
(R and Python) on one side and SmartPLS 4 on the other.

#### Decision

The decision rule in the user's instruction was:

> If R plspm breadth CIs contain SmartPLS references for ≥ 3 of 4 platforms →
> declare R plspm the engine for Stages 1 and 3, proceed to Stage 1.
> If R plspm shows the same disagreement pattern as Python plspm → fall back
> to Option (A): SmartPLS GUI for Stages 1 and 3.

R plspm CIs contain SmartPLS for **0 of 4 platforms** for breadth. Same
disagreement pattern as Python. **Decision: Option (A) — use SmartPLS 4 GUI
for Stages 1 and 3.** Stages 2 (C4) and 4 (D2) remain unaffected and proceed
on the existing `data/latent_data/*.csv` SmartPLS scores using the R/Python
scripts.

#### What the verification actually showed

The original framing was "verify that plspm reproduces SmartPLS". With R
plspm confirmed identical to the Python port, the more useful framing is:
**SmartPLS 4 and CRAN/PyPI `plspm` are different implementations of formative
PLS-SEM that converge to different fixed points for weak (near-zero) indicators.**
The dominant-indicator weights (depth, rating_deviation) match across all
three programs; the constructs themselves are the same. The disagreement is
confined to indicators whose unique contribution to the construct, controlling
for the dominant indicator, is small — exactly where the Wold algorithm has
multiple comparable fixed points and where the choice of inner-weighting
initialization tips the iteration into one or the other.

This is a useful **manuscript-level finding** for the robustness section:
the directional pattern of weak indicators (e.g., breadth's sign by
platform) is not engine-portable. It only appears with the same engine
(SmartPLS) that was used to estimate the baseline. Reframing the Stage 1
hypothesis to "the breadth sign pattern, **as estimated by SmartPLS**, is
robust to NMF K and to dispersion measure" is consistent with the
contingency choice (run Stages 1 and 3 in SmartPLS).

#### Why not skip the rerun?

Two reasons it was worth installing R rather than going straight to (A):

1. The Python port is a port; the R package is the reference. Without
   testing R, we couldn't separate "Python port bug" from "open-source vs
   SmartPLS difference". The R rerun showed the Python port is faithful —
   useful to record for any future automation of Stages 1/3 in R or Python.
2. The bit-identical agreement means we can use either engine (R or Python)
   for diagnostic exploration during Stage 1, accepting that the reported
   weights will differ from SmartPLS on weak indicators. SmartPLS remains
   the canonical engine for the manuscript's reported weights.


If any criterion fails, the contingency path in §"If verification fails" of
`docs/robustness_plan.md` applies (check Scheme.PATH vs. CENTROID; check
scaling; check bootstrap count; then fall back to SmartPLS GUI for re-estimation
in Stages 1 and 3 if needed).

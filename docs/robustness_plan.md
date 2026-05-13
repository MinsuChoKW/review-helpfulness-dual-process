# Robustness Checks: Execution Plan (v2)

This document specifies the robustness analyses to be added as a new Section §VI.D
in the manuscript ("Information Processing and Review Helpfulness in Online Platforms:
A Dual-Process Perspective").

The plan covers **one verification stage** followed by **four robustness checks
(A2, C4, D1, D2)**. Execution environment: VS Code with Claude Code,
using R (`plspm` for PLS-SEM, `glmmTMB` for ZINB/Multilevel ZINB) and Python
(for feature engineering of alternative breadth measures).

---

## Confirmed Baseline Specification

Both baseline R scripts in `r_scripts/` use the same fixed structure; they differ only
in the presence of random effects.

### Data inputs

```
../data/latent_data/{amazon,audible,coursera,hotel}.csv
```

Each CSV contains, at minimum:
- `Helpfulness`      — outcome (integer count)
- `Latent_Heuristic` — formative latent score for heuristic processing
- `Latent_Systematic` — formative latent score for systematic processing
- `Group`            — item-level grouping variable

Platform "hotel" corresponds to Booking.com in the manuscript.

### Standard ZINB (`zinb_modeling.R`)

```r
zinb_model <- glmmTMB(
  Helpfulness ~ Latent_Heuristic + Latent_Systematic,
  ziformula  = ~ Latent_Heuristic + Latent_Systematic,
  family     = nbinom2,
  data       = df
)
```

### Multilevel ZINB (`multilevel_zinb_modeling.R`)

```r
zinb_model <- glmmTMB(
  Helpfulness ~ Latent_Heuristic + Latent_Systematic
              + (Latent_Heuristic + Latent_Systematic | Group),
  ziformula  = ~ Latent_Heuristic + Latent_Systematic
              + (Latent_Heuristic + Latent_Systematic | Group),
  family     = nbinom2,
  data       = df
)
```

All robustness specifications below modify exactly one of:
(i) how `Latent_Heuristic` / `Latent_Systematic` are constructed (Stages 1, 3),
(ii) how the regression sample is defined (Stage 2),
(iii) how item-level confounders are absorbed (Stage 4).

### Note on manuscript Equation (11)

The manuscript's Equation (11) includes a `z_ij^T β_3` term for contextual covariates,
but the baseline code does not include covariates. This discrepancy should be resolved
at writing time (WP6): either drop the term from Eq. (11), or add a footnote clarifying
that contextual covariates were not used in the reported runs.

---

## Repository Conventions

Existing folders:
- `feature_engineering/`        — Python feature scripts
- `data/cleaned data/`          — preprocessed CSVs per platform
- `data/latent data/`           — SmartPLS-exported latent construct scores (baseline)
- `r_scripts/`                  — R scripts for baseline ZINB and Multilevel ZINB

New folders to create:
- `r_scripts/robustness/`       — R scripts for each robustness check
- `data/robustness/`            — intermediate outputs (alternative features and latent scores)
- `results/robustness/`         — per-spec coefficient tables
- `docs/`                       — this plan and notes

---

## Stage 0: Verification — Reproduce SmartPLS Baseline with R `plspm`

**Goal.** Confirm that R `plspm` reproduces the SmartPLS 4 baseline indicator weights
(Tables 2 and 3 of the manuscript) within acceptable tolerance. Once verified, `plspm`
becomes the engine for all subsequent PLS re-estimations (Stages 1 and 3), enabling
full automation inside VS Code.

`plspm` is already imported in both baseline R scripts (although not actively used there),
so no new dependency is added.

### Inputs

- `data/cleaned data/*.csv` — preprocessed reviews per platform, containing the seven
  observed indicators:
    - Systematic: `depth`, `breadth`, `readability`, `arousal`
    - Heuristic:  `rating_deviation`, `title_length`, `recency`
    - (Coursera lacks `title_length`)
- Baseline indicator weights from manuscript Tables 2 and 3.

### Steps

1. Install `plspm` if not present: `install.packages("plspm")`.
2. For each platform, build a `plspm` model:
    - Two formative blocks: `Systematic` and `Heuristic`.
    - Outer modes: `"B"` (formative) for both blocks.
    - Inner model: a unidirectional path from each construct to a placeholder outcome
      block. `plspm` requires an inner model with at least one path; see Open
      Questions for the recommended choice of placeholder.
3. Run `plspm()` with `scheme = "path"` and bootstrap (`boot.val = TRUE`, `br = 1000`).
4. Extract standardized indicator weights and bootstrap p-values.
5. Compare to manuscript Tables 2 and 3.

### Success criteria

- Indicator weights match the manuscript's reported weights within **±0.03** in
  absolute value.
- Sign of every weight matches.
- Significance category (*, **, ***, n.s.) matches in at least **22 out of 24** cells
  (4 platforms × ≤6 indicators).

### If verification fails

Diagnose in this order:
1. Check whether `plspm` uses centroid vs. path weighting (default centroid; SmartPLS
   default is path — must set `scheme = "path"`).
2. Check standardization (`scaled = TRUE` is the `plspm` default).
3. Compare bootstrap replication counts.
If discrepancies remain, see the contingency plan at the end of this document.

### Output

- `results/robustness/stage0_plspm_vs_smartpls.csv` — side-by-side weights per platform.
- `docs/stage0_verification_notes.md` — brief written assessment.

**Do not proceed to Stages 1–4 until Stage 0 has been signed off.**

---

## Stage 1: A2 — Alternative Measurement of Breadth

**Reviewer link.** Reviewer 2 #1 (H1b reframing as conditional);
Reviewer 3 #1.3 (breadth inconsistency); Reviewer 3 #2 (robustness checks).

**Goal.** Show that the directional pattern of breadth — positive in Booking.com,
negative in the other three platforms — is not an artifact of (i) the number of topics
chosen for NMF or (ii) the use of KL divergence as the dispersion measure.

### Specifications

| Sub-spec | Description |
|----------|-------------|
| A2-a     | NMF with K = 5 (smaller than baseline) |
| A2-b     | NMF with K = 15 (larger than baseline) |
| A2-c     | Replace KL divergence with topic entropy: `Breadth_j = -Σ p_ij log p_ij` |

### Steps

1. In `feature_engineering/`, locate the existing breadth script. Confirm the baseline K.
2. For each sub-spec, generate `breadth_<spec>.csv` per platform.
3. Merge with existing values for depth, readability, arousal, and all heuristic
   indicators (unchanged).
4. Run `plspm` PLS-SEM per platform per sub-spec → write latent scores
   (`Latent_Heuristic`, `Latent_Systematic`, `Group`, `Helpfulness`) to
   `data/robustness/latent_A2-<sub>/<platform>.csv`.
5. Run Multilevel ZINB (`glmmTMB`, exactly as baseline) on each new latent-score file.
6. Extract the four core coefficients per platform per sub-spec:
   - conditional `Latent_Systematic`
   - conditional `Latent_Heuristic`
   - zero-inflation `Latent_Systematic`
   - zero-inflation `Latent_Heuristic`

### Reporting

- Per platform, sign and significance across baseline, A2-a, A2-b, A2-c.
- Sign-concordance rate vs. baseline.
- Breadth weight in the systematic construct under each sub-spec.

### Expected pattern

- Conditional `Systematic` and `Heuristic` coefficients: signs match baseline in all
  sub-specs and platforms.
- Breadth weight remains positive on Booking.com, negative on Amazon/Audible/Coursera.

### Output

- `results/robustness/A2_breadth_alternative.csv` in long format with columns
  `(platform, sub_spec, coefficient, estimate, se, p_value)`.

---

## Stage 2: C4 — Exposure-Conditional Analysis on Read-Review Subsample

**Reviewer link.** Reviewer 2 #5 (exposure mechanism); Reviewer 3 #2 (exposure bias);
internal link to §VII.B (Limitations) and §III.A.2 (recency justification).

**Goal.** If the ZINB's two-stage separation is well calibrated, restricting the sample
to reviews that received at least one helpful vote (i.e., known to have been exposed)
should yield conditional-component coefficients similar to the baseline Multilevel ZINB
conditional component.

### Specification

| Sub-spec | Description |
|----------|-------------|
| C4 | Multilevel NB2 on `{rows with Helpfulness > 0}`. Same fixed and random structure as baseline, but without a zero-inflation component. |

### Steps

1. Use existing `data/latent_data/*.csv` (no PLS re-estimation needed).
2. Create `r_scripts/robustness/c4_exposed_subsample.R`:
    ```r
    df_pos <- df %>% filter(Helpfulness > 0)
    nb2_model <- glmmTMB(
      Helpfulness ~ Latent_Heuristic + Latent_Systematic
                  + (Latent_Heuristic + Latent_Systematic | Group),
      family = nbinom2,
      data   = df_pos
    )
    ```
3. Per platform, extract the two conditional coefficients.
4. Compare to baseline Multilevel ZINB conditional coefficients (manuscript Table 6).

### Reporting

- Per platform: baseline `Systematic`, `Heuristic` (conditional) vs. C4 NB2 estimates,
  with subsample size `n` and proportion of full sample retained.
- Relative difference: `|β_C4 − β_baseline| / |β_baseline|`.

### Expected pattern

- Signs identical to baseline.
- Magnitudes within ±30% of baseline.
- If random slopes become unidentifiable on the reduced sample, fall back to a
  random-intercept-only specification and note it.

### Output

- `results/robustness/C4_exposed_subsample.csv`.

---

## Stage 3: D1 — Recency Removed from Heuristic Construct

**Reviewer link.** Reviewer 2 #2 (recency as heuristic vs. visibility proxy);
internal link to WP3-b expansion of §III.A.2.

**Goal.** Show that the helpfulness effect of the heuristic construct does not depend
on the inclusion of recency. If the construct remains significantly positive after
recency is removed, this supports the cognitive-heuristic reading of the remaining
indicators and weakens the alternative "recency = pure visibility proxy" interpretation.

### Specification

| Sub-spec | Description |
|----------|-------------|
| D1 | Re-estimate the Heuristic block in `plspm` without `recency`. Everything else identical. |

### Indicator lists under D1

- Amazon, Audible, hotel (Booking.com): Heuristic = `{rating_deviation, title_length}`
- Coursera: Heuristic = `{rating_deviation}` only.
  Coursera's Heuristic becomes a single-indicator formative block, which is a valid but
  edge specification. Flag this in the manuscript as a caveat — the Coursera D1 result
  is suggestive rather than decisive.

### Steps

1. Build modified `plspm` model with the reduced heuristic block.
2. Re-estimate per platform; write latent scores to
   `data/robustness/latent_D1/<platform>.csv`.
3. Run Multilevel ZINB on each new latent-score file.
4. Extract the four core coefficients per platform.

### Reporting

- Per platform: baseline vs. D1 for `Heuristic` (conditional) and `Heuristic`
  (zero-inflation).
- Whether the H3b conclusion (Heuristic → Helpfulness) holds.

### Expected pattern

- Heuristic-conditional remains significantly positive in at least 3 of 4 platforms.
- Heuristic-zero-inflation remains significantly negative in Amazon, Audible, Coursera;
  Booking.com may retain its positive sign — itself informative.

### Output

- `results/robustness/D1_recency_removed.csv`.

---

## Stage 4: D2 — Item-Level Fixed Effects (Within-Item Centering)

**Reviewer link.** Reviewer 3 #2 (endogeneity); Reviewer 3 #1.3 (platform-specific
results).

**Goal.** Show that baseline Multilevel ZINB findings are not driven by unobserved
time-invariant item characteristics. Baseline already includes item-level random
effects; D2 strengthens this by item-mean centering, which removes between-item
variation from fixed-effect identification.

### Specification

| Sub-spec | Description |
|----------|-------------|
| D2 | Item-mean centering of `Latent_Heuristic` and `Latent_Systematic` before fitting. Random structure retained. |

### Steps

1. Use existing `data/latent_data/*.csv`.
2. Compute within-item-centered constructs:
    ```r
    df <- df %>%
      group_by(Group) %>%
      mutate(
        Latent_Heuristic_wc  = Latent_Heuristic  - mean(Latent_Heuristic),
        Latent_Systematic_wc = Latent_Systematic - mean(Latent_Systematic)
      ) %>%
      ungroup()
    ```
3. Fit Multilevel ZINB with the within-centered variables as fixed effects, retaining
   the original random structure.

### Reporting

- Per platform: baseline vs. D2 for all four core coefficients.
- Note that D2 coefficients reflect *within-item* effects only.

### Expected pattern

- Signs match baseline.
- Magnitudes may shrink (between-item variation excluded).
- If signs flip or significance disappears, report transparently — would indicate that
  between-item differences drive part of the baseline result.

### Output

- `results/robustness/D2_item_FE.csv`.

---

## Stage 5: Aggregation and Manuscript Table

### Steps

1. Build `results/robustness/summary_table.csv`:

    | Specification     | Platform     | Sys-Cond  | Heu-Cond  | Sys-ZI    | Heu-ZI    |
    |-------------------|--------------|-----------|-----------|-----------|-----------|
    | Baseline          | Amazon       | 0.717***  | 0.134***  | -1.549*** | -1.468*** |
    | A2-a (NMF K=5)    | Amazon       | ...       | ...       | ...       | ...       |
    | A2-b (NMF K=15)   | Amazon       | ...       | ...       | ...       | ...       |
    | A2-c (entropy)    | Amazon       | ...       | ...       | ...       | ...       |
    | C4 (helpful>0)    | Amazon       | ...       | ...       | (n/a)     | (n/a)     |
    | D1 (no recency)   | Amazon       | ...       | ...       | ...       | ...       |
    | D2 (item FE)      | Amazon       | ...       | ...       | ...       | ...       |
    | ... (other platforms follow)                                                          |

2. For the manuscript body (Table 9), present a compact version: per specification,
   count of (platform × coefficient) cells where the sign matches baseline, and where
   the significance level matches.

3. For Appendix B, the full table.

### Output

- `results/robustness/summary_table.csv` (full)
- `results/robustness/main_table9.csv` (compact for manuscript body)
- `results/robustness/appendix_B_detail.csv` (detailed for appendix)

---

## Contingency: If `plspm` Fails Verification

Fall back to a hybrid plan:
- Use SmartPLS 4 GUI for Stages 1 and 3 (the stages requiring re-estimation), accepting
  the GUI overhead (16 SmartPLS runs total: A2 × 3 sub-specs × 4 platforms +
  D1 × 4 platforms).
- Stages 2 and 4 remain R-only and unaffected.
- Alternative: reduce A2 to a single sub-spec (e.g., A2-a only) to cut the SmartPLS
  load by two-thirds.

---

## Estimated Workload

| Stage | Approximate effort |
|-------|--------------------|
| Stage 0 (verification)        | 1 session — required before anything else |
| Stage 1 (A2, 3 sub-specs)     | 2 sessions — full pipeline per sub-spec × 4 platforms |
| Stage 2 (C4)                  | 0.5 session — R only |
| Stage 3 (D1)                  | 1 session — PLS re-estimation per platform |
| Stage 4 (D2)                  | 0.5 session — R only |
| Stage 5 (aggregation)         | 0.5 session |

Stages 2 and 4 can run in parallel since they need no PLS re-estimation. Stages 1 and 3
require PLS and should be sequential.

---

## Open Questions to Resolve Before Stage 0

1. Confirm the exact NMF K value used in the baseline breadth script
   (`feature_engineering/`).
2. Confirm whether `Helpfulness`, `Latent_*`, and `Group` columns in
   `data/latent_data/*.csv` use any nonstandard scaling or transformation that should
   be replicated in `plspm` reruns.
3. Confirm the synthetic outcome strategy for `plspm` inner model (since `plspm`
   requires an inner model). Two natural choices:
   - (a) Use `Helpfulness` itself as a one-indicator outcome construct.
   - (b) Use a placeholder construct whose indicator is `log1p(Helpfulness)`.
   Either is fine for *measurement* validation since we only care about outer weights.

Document the answers in `docs/baseline_specification.md` before Stage 0 begins.

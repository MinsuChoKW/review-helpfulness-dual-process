#!/usr/bin/env Rscript
# Multi-stage robustness driver: A2 (3 sub-specs) + D1 + C4 + D2.
#
# Reads SmartPLS-exported latent scores for A2 and D1 sub-specs, attaches the
# canonical Group ID from the sanitized input CSV (SmartPLS preserves row
# order so we merge by position), renames to baseline conventions, and fits
# the baseline Multilevel ZINB. Falls back to random-intercept-only on
# non-PD Hessian (random-slopes singular). Adds Stage 2 (C4) NB2 on the
# Helpfulness>0 subsample and Stage 4 (D2) within-item-centered ZINB.
#
# Baseline comparator is manuscript Tables 2 (outer weights for breadth) and
# 6 (Multilevel ZINB coefficients). The baseline was not re-run via SmartPLS
# in this revision — to be flagged with a manuscript footnote at write-up.

suppressPackageStartupMessages({
  library(dplyr)
  library(glmmTMB)
})

REPO         <- "/Users/minsucho/Documents/Helpfulness/revisions"
LATENT_BASE  <- file.path(REPO, "smartpls/results/latent_score")
WEIGHT_BASE  <- file.path(REPO, "smartpls/results/outer_weight")
SP_INPUT_BASE<- file.path(REPO, "data/robustness/smartpls_input")
BASELINE_LAT <- file.path(REPO, "data/latent_data")
OUT_DIR      <- file.path(REPO, "results/robustness")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

PLATFORMS <- c("amazon", "audible", "coursera", "hotel")
A2_D1     <- c("a2-a", "a2-b", "a2-c", "d1")

# Manuscript Table 6 baseline coefficients
BASELINE <- list(
  amazon   = list(cond_Sys=0.717, cond_Sys_sig="***",
                  cond_Heu=0.134, cond_Heu_sig="***",
                  zi_Sys  =-1.549, zi_Sys_sig ="***",
                  zi_Heu  =-1.468, zi_Heu_sig ="***"),
  hotel    = list(cond_Sys=0.188, cond_Sys_sig="***",
                  cond_Heu=0.087, cond_Heu_sig="**",
                  zi_Sys  =-1.713, zi_Sys_sig ="***",
                  zi_Heu  = 0.140, zi_Heu_sig ="***"),
  audible  = list(cond_Sys=0.694, cond_Sys_sig="***",
                  cond_Heu=1.303, cond_Heu_sig="***",
                  zi_Sys  = 0.080, zi_Sys_sig ="***",
                  zi_Heu  =-1.164, zi_Heu_sig ="***"),
  coursera = list(cond_Sys=0.231, cond_Sys_sig="***",
                  cond_Heu=0.132, cond_Heu_sig="***",
                  zi_Sys  =-1.009, zi_Sys_sig ="***",
                  zi_Heu  =-1.070, zi_Heu_sig ="***")
)

# Manuscript Table 2 baseline breadth outer weights
BASELINE_BREADTH <- list(
  amazon   = list(w=-0.085, sig="***"),
  hotel    = list(w= 0.251, sig="***"),
  audible  = list(w=-0.108, sig="***"),
  coursera = list(w=-0.159, sig="***")
)

stars <- function(p) {
  if (is.na(p)) return("")
  if (p < 0.001) return("***")
  if (p < 0.01)  return("**")
  if (p < 0.05)  return("*")
  "n.s."
}

# Load SmartPLS-exported latent scores for a (sub-spec, platform), strip the
# UTF-8 BOM, attach Group by row order from the sanitized input CSV, and
# rename to baseline conventions.
load_latent <- function(sub_spec, platform) {
  # SmartPLS export dirs use lowercase sub-spec names (a2-a, d1);
  # sanitized input dirs use the mixed-case names (A2-a, D1) but the
  # sanitized filenames carry the lowercase sub-spec prefix
  # (e.g. A2-a/a2-a-amazon_smartpls.csv, D1/d1-amazon_smartpls.csv).
  sub_spec_input_dir <- switch(sub_spec,
    "a2-a" = "A2-a",
    "a2-b" = "A2-b",
    "a2-c" = "A2-c",
    "d1"   = "D1"
  )
  latent_path <- file.path(LATENT_BASE, sub_spec,
                           sprintf("%s_%s.csv", sub_spec, platform))
  input_path  <- file.path(SP_INPUT_BASE, sub_spec_input_dir,
                           sprintf("%s-%s_smartpls.csv", sub_spec, platform))
  latent <- read.csv(latent_path, fileEncoding = "UTF-8-BOM")
  inp    <- read.csv(input_path)
  if (nrow(latent) != nrow(inp)) {
    stop(sprintf("Row count mismatch: latent=%d, input=%d at %s/%s",
                 nrow(latent), nrow(inp), sub_spec, platform))
  }
  # SmartPLS standardizes every manifest variable by default, including the
  # outcome's pass-through Helpfulness indicator. The ZINB outcome must be
  # the raw integer count, so we always take Helpfulness from the input CSV
  # (which holds raw counts), not from the SmartPLS export.
  out <- data.frame(
    Helpfulness       = inp$Helpfulness,
    Group             = inp$Group,
    Latent_Heuristic  = latent$Heuristic,
    Latent_Systematic = latent$Systematic
  )
  out
}

# Multilevel ZINB with random-slopes; fallback to random-intercept-only on
# non-PD Hessian. Set drop_zi = TRUE to fit NB2 only (used for Stage C4).
fit_zinb <- function(df, drop_zi = FALSE) {
  used_fallback <- FALSE
  err_msg <- ""

  fit <- tryCatch({
    if (drop_zi) {
      glmmTMB(
        Helpfulness ~ Latent_Heuristic + Latent_Systematic
                    + (Latent_Heuristic + Latent_Systematic | Group),
        family = nbinom2, data = df
      )
    } else {
      glmmTMB(
        Helpfulness ~ Latent_Heuristic + Latent_Systematic
                    + (Latent_Heuristic + Latent_Systematic | Group),
        ziformula = ~ Latent_Heuristic + Latent_Systematic
                    + (Latent_Heuristic + Latent_Systematic | Group),
        family = nbinom2, data = df
      )
    }
  }, error = function(e) {
    err_msg <<- paste("rs-error:", conditionMessage(e))
    NULL
  })

  conv_ok <- !is.null(fit) && !is.null(fit$sdr$pdHess) && fit$sdr$pdHess
  if (!conv_ok) {
    used_fallback <- TRUE
    if (nchar(err_msg) == 0) err_msg <- "random-slopes Hessian non-PD"
    fit <- tryCatch({
      if (drop_zi) {
        glmmTMB(
          Helpfulness ~ Latent_Heuristic + Latent_Systematic + (1 | Group),
          family = nbinom2, data = df
        )
      } else {
        glmmTMB(
          Helpfulness ~ Latent_Heuristic + Latent_Systematic + (1 | Group),
          ziformula = ~ Latent_Heuristic + Latent_Systematic + (1 | Group),
          family = nbinom2, data = df
        )
      }
    }, error = function(e) {
      err_msg <<- paste(err_msg, "; RI-error:", conditionMessage(e))
      NULL
    })
  }

  if (is.null(fit)) stop("both ZINB variants failed: ", err_msg)
  list(model = fit, used_fallback = used_fallback, err_msg = err_msg,
       conv_ok = !is.null(fit$sdr$pdHess) && fit$sdr$pdHess)
}

extract_coefs <- function(fit_res, spec, platform, drop_zi = FALSE) {
  s <- summary(fit_res$model)
  cs_cond <- s$coefficients$cond
  cs_zi   <- if (!drop_zi) s$coefficients$zi else NULL
  rows <- list()
  components <- if (drop_zi) "conditional" else c("conditional", "zero_inflation")
  for (component in components) {
    cs <- if (component == "conditional") cs_cond else cs_zi
    for (construct in c("Systematic", "Heuristic")) {
      term <- paste0("Latent_", construct)
      if (term %in% rownames(cs)) {
        est <- cs[term, "Estimate"]
        se  <- cs[term, "Std. Error"]
        p   <- cs[term, "Pr(>|z|)"]
      } else {
        est <- NA; se <- NA; p <- NA
      }
      conv_status <- if (fit_res$used_fallback) "fallback_RI" else if (fit_res$conv_ok) "OK" else "warn"
      rows[[length(rows) + 1]] <- data.frame(
        spec        = spec,
        platform    = platform,
        component   = component,
        construct   = construct,
        estimate    = round(est, 4),
        se          = round(se, 4),
        p_value     = round(p, 6),
        sig_marker  = stars(p),
        conv_status = conv_status,
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, rows)
}

# ==== Section 1: A2 and D1 (16 ZINB fits) ====

all_coefs <- list()
conv_log  <- list()

for (sub_spec in A2_D1) {
  for (platform in PLATFORMS) {
    cat(sprintf("\n=== %s / %s ===\n", sub_spec, platform))
    df <- load_latent(sub_spec, platform)
    cat(sprintf("  N=%d  unique Group=%d\n", nrow(df), length(unique(df$Group))))
    fit_res <- fit_zinb(df, drop_zi = FALSE)
    cat(sprintf("  used_fallback=%s  conv_ok=%s\n",
                fit_res$used_fallback, fit_res$conv_ok))
    coefs <- extract_coefs(fit_res, sub_spec, platform, drop_zi = FALSE)
    all_coefs[[length(all_coefs) + 1]] <- coefs
    conv_log[[length(conv_log) + 1]] <- data.frame(
      spec = sub_spec, platform = platform,
      original_status = if (fit_res$used_fallback) "non-PD-Hessian-or-rs-error" else "OK",
      fallback_used   = as.integer(fit_res$used_fallback),
      final_status    = if (fit_res$conv_ok) "OK" else "warn",
      err_msg         = fit_res$err_msg,
      stringsAsFactors = FALSE
    )
    print(coefs)
  }
}

# ==== Section 2: Stage C4 (Multilevel NB2 on Helpfulness > 0, baseline scores) ====

cat("\n\n=========== Stage 2 (C4) ===========\n")
for (platform in PLATFORMS) {
  cat(sprintf("\n=== c4 / %s ===\n", platform))
  df <- read.csv(file.path(BASELINE_LAT, paste0(platform, ".csv")))
  df_pos <- df %>% filter(Helpfulness > 0)
  cat(sprintf("  N_full=%d  N_pos=%d  retention=%.4f\n",
              nrow(df), nrow(df_pos), nrow(df_pos)/nrow(df)))
  fit_res <- fit_zinb(df_pos, drop_zi = TRUE)
  cat(sprintf("  used_fallback=%s  conv_ok=%s\n",
              fit_res$used_fallback, fit_res$conv_ok))
  coefs <- extract_coefs(fit_res, "c4", platform, drop_zi = TRUE)
  all_coefs[[length(all_coefs) + 1]] <- coefs
  conv_log[[length(conv_log) + 1]] <- data.frame(
    spec = "c4", platform = platform,
    original_status = if (fit_res$used_fallback) "non-PD-Hessian-or-rs-error" else "OK",
    fallback_used   = as.integer(fit_res$used_fallback),
    final_status    = if (fit_res$conv_ok) "OK" else "warn",
    err_msg         = fit_res$err_msg,
    stringsAsFactors = FALSE
  )
  print(coefs)
}

# ==== Section 3: Stage D2 (within-item-centered Multilevel ZINB) ====

cat("\n\n=========== Stage 4 (D2) ===========\n")
for (platform in PLATFORMS) {
  cat(sprintf("\n=== d2 / %s ===\n", platform))
  df <- read.csv(file.path(BASELINE_LAT, paste0(platform, ".csv")))
  df <- df %>%
    group_by(Group) %>%
    mutate(
      Latent_Heuristic  = Latent_Heuristic  - mean(Latent_Heuristic),
      Latent_Systematic = Latent_Systematic - mean(Latent_Systematic)
    ) %>%
    ungroup()
  cat(sprintf("  N=%d  unique Group=%d\n", nrow(df), length(unique(df$Group))))
  fit_res <- fit_zinb(df, drop_zi = FALSE)
  cat(sprintf("  used_fallback=%s  conv_ok=%s\n",
              fit_res$used_fallback, fit_res$conv_ok))
  coefs <- extract_coefs(fit_res, "d2", platform, drop_zi = FALSE)
  all_coefs[[length(all_coefs) + 1]] <- coefs
  conv_log[[length(conv_log) + 1]] <- data.frame(
    spec = "d2", platform = platform,
    original_status = if (fit_res$used_fallback) "non-PD-Hessian-or-rs-error" else "OK",
    fallback_used   = as.integer(fit_res$used_fallback),
    final_status    = if (fit_res$conv_ok) "OK" else "warn",
    err_msg         = fit_res$err_msg,
    stringsAsFactors = FALSE
  )
  print(coefs)
}

# ==== Section 4: write the long-format coefficient table ====

all_coefs_df <- do.call(rbind, all_coefs)
out_long <- file.path(OUT_DIR, "all_specs_coefficients.csv")
write.csv(all_coefs_df, out_long, row.names = FALSE)
cat(sprintf("\nWrote %s (%d rows)\n", out_long, nrow(all_coefs_df)))

conv_df <- do.call(rbind, conv_log)
out_conv <- file.path(OUT_DIR, "convergence_log.csv")
write.csv(conv_df, out_conv, row.names = FALSE)
cat(sprintf("Wrote %s (%d rows)\n", out_conv, nrow(conv_df)))

# ==== Section 5: summary_table.csv (sign + sig concordance per spec) ====

build_summary <- function(coef_df, spec_filter) {
  rows <- list()
  total_cells <- 0; total_sign <- 0; total_sig <- 0
  for (platform in PLATFORMS) {
    base <- BASELINE[[platform]]
    sub  <- coef_df %>% filter(spec == spec_filter, platform == !!platform)
    n_sign_p <- 0; n_sig_p <- 0; n_cells_p <- 0
    components <- if (spec_filter == "c4") "conditional" else c("conditional", "zero_inflation")
    for (component in components) {
      for (construct in c("Systematic", "Heuristic")) {
        key_est <- if (component == "conditional") {
          if (construct == "Systematic") "cond_Sys" else "cond_Heu"
        } else {
          if (construct == "Systematic") "zi_Sys"   else "zi_Heu"
        }
        key_sig <- paste0(key_est, "_sig")
        base_est <- base[[key_est]]; base_sig <- base[[key_sig]]
        spec_row <- sub %>% filter(component == !!component, construct == !!construct)
        if (nrow(spec_row) == 0) next
        spec_est <- spec_row$estimate[1]; spec_sig <- spec_row$sig_marker[1]
        sign_m <- if (!is.na(spec_est) && !is.na(base_est)) as.integer(sign(spec_est) == sign(base_est)) else 0
        sig_m  <- if (!is.na(spec_sig) && nchar(spec_sig) > 0 && nchar(base_sig) > 0) as.integer(spec_sig == base_sig) else 0
        n_sign_p  <- n_sign_p  + sign_m
        n_sig_p   <- n_sig_p   + sig_m
        n_cells_p <- n_cells_p + 1
      }
    }
    rows[[length(rows)+1]] <- data.frame(
      spec = spec_filter, platform = platform,
      n_cells = n_cells_p, n_sign_match = n_sign_p, n_sig_match = n_sig_p,
      stringsAsFactors = FALSE
    )
    total_cells <- total_cells + n_cells_p
    total_sign  <- total_sign  + n_sign_p
    total_sig   <- total_sig   + n_sig_p
  }
  per_platform <- do.call(rbind, rows)
  overall <- data.frame(
    spec = spec_filter, platform = "TOTAL",
    n_cells = total_cells, n_sign_match = total_sign,
    n_sig_match = total_sig, stringsAsFactors = FALSE
  )
  rbind(per_platform, overall)
}

summary_parts <- list()
for (sf in c("a2-a", "a2-b", "a2-c", "d1", "c4", "d2")) {
  summary_parts[[length(summary_parts)+1]] <- build_summary(all_coefs_df, sf)
}
summary_df <- do.call(rbind, summary_parts)
summary_df$footnote <- ifelse(
  summary_df$spec == "d1" & summary_df$platform == "coursera",
  "Heuristic under D1 is single-indicator (RatingDeviation only); coefficient reflects RatingDeviation effect alone.",
  ""
)
write.csv(summary_df, file.path(OUT_DIR, "summary_table.csv"), row.names = FALSE)
cat("Wrote summary_table.csv\n")

# ==== Section 6: A2 breadth outer-weight comparison ====

parse_outer_weight <- function(path, target_row) {
  ow <- read.csv(path, fileEncoding = "UTF-8-BOM", stringsAsFactors = FALSE)
  colnames(ow)[1] <- "path"
  row <- ow[ow$path == target_row, ]
  if (nrow(row) == 0) return(list(w = NA, p = NA, sig = ""))
  w <- as.numeric(row[["Original.sample..O."]][1])
  p_str <- row[["P.values"]][1]
  p <- suppressWarnings(as.numeric(p_str))
  list(w = w, p = p, sig = stars(p))
}

a2_rows <- list()
for (sub_spec in c("a2-a", "a2-b", "a2-c")) {
  for (platform in PLATFORMS) {
    path <- file.path(WEIGHT_BASE, sub_spec,
                      sprintf("%s_%s.csv", sub_spec, platform))
    pw <- parse_outer_weight(path, "Breadth -> Systematic")
    base <- BASELINE_BREADTH[[platform]]
    a2_rows[[length(a2_rows)+1]] <- data.frame(
      sub_spec      = sub_spec,
      platform      = platform,
      baseline_weight = base$w,
      baseline_sig    = base$sig,
      new_weight    = round(pw$w, 4),
      new_sig       = pw$sig,
      new_p_value   = round(pw$p, 6),
      weight_diff   = round(pw$w - base$w, 4),
      sign_match    = if (!is.na(pw$w)) as.integer(sign(pw$w) == sign(base$w)) else 0,
      stringsAsFactors = FALSE
    )
  }
}
a2_breadth_df <- do.call(rbind, a2_rows)
write.csv(a2_breadth_df, file.path(OUT_DIR, "A2_breadth_weights.csv"), row.names = FALSE)
cat("Wrote A2_breadth_weights.csv\n")

# ==== Section 7: D1 heuristic-coefficient comparison ====

d1_rows <- list()
for (platform in PLATFORMS) {
  base <- BASELINE[[platform]]
  sub <- all_coefs_df %>%
    filter(spec == "d1", platform == !!platform, construct == "Heuristic")
  cond <- sub %>% filter(component == "conditional")
  zi   <- sub %>% filter(component == "zero_inflation")
  d1_rows[[length(d1_rows)+1]] <- data.frame(
    platform = platform,
    baseline_cond_Heu = base$cond_Heu,
    baseline_cond_Heu_sig = base$cond_Heu_sig,
    d1_cond_Heu = if (nrow(cond) > 0) cond$estimate[1] else NA,
    d1_cond_Heu_sig = if (nrow(cond) > 0) cond$sig_marker[1] else "",
    cond_diff = if (nrow(cond) > 0) round(cond$estimate[1] - base$cond_Heu, 4) else NA,
    cond_sign_match = if (nrow(cond) > 0 && !is.na(cond$estimate[1])) as.integer(sign(cond$estimate[1]) == sign(base$cond_Heu)) else 0,
    baseline_zi_Heu = base$zi_Heu,
    baseline_zi_Heu_sig = base$zi_Heu_sig,
    d1_zi_Heu = if (nrow(zi) > 0) zi$estimate[1] else NA,
    d1_zi_Heu_sig = if (nrow(zi) > 0) zi$sig_marker[1] else "",
    zi_diff = if (nrow(zi) > 0) round(zi$estimate[1] - base$zi_Heu, 4) else NA,
    zi_sign_match = if (nrow(zi) > 0 && !is.na(zi$estimate[1])) as.integer(sign(zi$estimate[1]) == sign(base$zi_Heu)) else 0,
    caveat = if (platform == "coursera") "Heuristic under D1 is single-indicator (RatingDeviation only)" else "",
    stringsAsFactors = FALSE
  )
}
d1_df <- do.call(rbind, d1_rows)
write.csv(d1_df, file.path(OUT_DIR, "D1_no_recency_heuristic.csv"), row.names = FALSE)
cat("Wrote D1_no_recency_heuristic.csv\n")

cat("\n=== ALL DONE ===\n")

# Step 07: measurement invariance of five scales, open-data university against KNUST.
# Run:     Rscript 04_analysis/07_invariance.R        (from the repository folder)
# Needs:   R 4.2 or later with lavaan and semTools (install.packages(c("lavaan", "semTools")))
# Output:  05_results/tables/T4_invariance.csv
# Steps:   configural -> metric (equal loadings) -> scalar (equal loadings and intercepts).
#          If scalar fails, the intercept with the largest score test is freed (partial scalar).
# Rule:    a step holds if CFI drops by 0.01 or less and RMSEA rises by 0.015 or less (Chen, 2007).

suppressPackageStartupMessages({ library(lavaan); library(semTools) })

settings <- readLines(file.path("04_analysis", "settings.py"))
mode <- sub('^DATA_MODE\\s*=\\s*"([a-z]+)".*', "\\1", grep("^DATA_MODE", settings, value = TRUE))
cat("Step 07: invariance. Mode:", toupper(mode), "\n")
if (mode == "rehearsal") {
  open <- read.csv(file.path("03_data", "rehearsal", "REHEARSAL_staff.csv"))
  open <- open[open$rule_published == 1, ]
  knust <- read.csv(file.path("03_data", "rehearsal", "REHEARSAL_knust.csv"))
  out_name <- "REHEARSAL_T4_invariance.csv"
  cat("SYNTHETIC DEMO DATA - not a result\n")
} else {
  open <- read.csv(file.path("03_data", "clean", "staff_clean.csv"))
  open <- open[open$rule_published == 1, ]
  kpath <- file.path("03_data", "clean", "knust_clean.csv")
  if (!file.exists(kpath)) stop("Missing ", kpath, ". Fix: run 04_analysis/02b_import_knust.py first.")
  knust <- read.csv(kpath)
  out_name <- "T4_invariance.csv"
}

scales <- list(
  optimism_bias      = "self =~ ob_self1 + ob_self2\n peer =~ ob_peer1 + ob_peer2",
  self_efficacy      = "f =~ se1 + se2 + se3 + se4",
  risk_perception    = "f =~ rp1 + rp2 + rp3 + rp4 + rp6",
  locus_of_control   = paste("f =~", paste0("lc", 1:12, collapse = " + ")),
  training_intention = "f =~ ti1 + ti2 + ti3 + ti4 + ti5"
)

fitm <- function(model, data, equal, partial = character(0)) {
  suppressWarnings(cfa(model, data = data, group = "site", group.equal = equal, group.partial = partial,
                       estimator = "MLR", missing = "ml"))
}
fi <- function(f) fitMeasures(f, c("cfi.robust", "rmsea.robust", "srmr"))

rows <- list()
for (nm in names(scales)) {
  items <- trimws(unlist(strsplit(gsub(".*=~", "", unlist(strsplit(scales[[nm]], "\n"))), "\\+")))
  dat <- rbind(open[, c("site", items)], knust[, c("site", items)])
  dat <- dat[rowSums(!is.na(dat[, items])) > 0, ]
  conf <- fitm(scales[[nm]], dat, "")
  metr <- fitm(scales[[nm]], dat, "loadings")
  scal <- fitm(scales[[nm]], dat, c("loadings", "intercepts"))
  f0 <- fi(conf); f1 <- fi(metr); f2 <- fi(scal)
  ok <- function(a, b) (a[1] - b[1]) <= 0.01 && (b[2] - a[2]) <= 0.015
  freed <- "none"
  if (!ok(f1, f2)) {                                  # partial scalar: free the worst intercept
    st <- suppressWarnings(lavTestScore(scal))$uni
    pt <- parTable(scal)
    is_intercept <- sapply(st$lhs, function(l) pt$op[pt$plabel == l][1] == "~1")
    st <- st[is_intercept, ]                          # only intercept constraints may be freed
    worst <- st[which.max(st$X2), ]
    freed <- paste0(pt$lhs[pt$plabel == worst$lhs][1], "~1")
    scal <- fitm(scales[[nm]], dat, c("loadings", "intercepts"), partial = freed)
    f2 <- fi(scal)
  }
  n_open <- sum(dat$site == "open_data"); n_kn <- sum(dat$site == "knust")
  rows[[nm]] <- data.frame(
    scale = nm, n_open_data = n_open, n_knust = n_kn,
    cfi_configural = round(f0[1], 3), rmsea_configural = round(f0[2], 3), srmr_configural = round(f0[3], 3),
    d_cfi_metric = round(f1[1] - f0[1], 3), d_rmsea_metric = round(f1[2] - f0[2], 3),
    metric_holds = ok(f0, f1),
    d_cfi_scalar = round(f2[1] - f1[1], 3), d_rmsea_scalar = round(f2[2] - f1[2], 3),
    scalar_holds = ok(f1, f2), intercept_freed = freed,
    solution_ok = isTRUE(suppressWarnings(lavInspect(conf, "post.check"))))
}
res <- do.call(rbind, rows)
print(res, row.names = FALSE)
if (any(!res$solution_ok)) cat("\nNOTE: solution_ok = FALSE means a variance or correlation is out of range for that scale.",
                               "\nSee Appendix A, A16, before you interpret that row.\n")
dir.create(file.path("05_results", "tables"), recursive = TRUE, showWarnings = FALSE)
write.csv(res, file.path("05_results", "tables", out_name), row.names = FALSE)
cat("  table -> 05_results/tables/", out_name, "\n", sep = "")
cat("\nDone. Next: python 04_analysis/08_figures.py\n")

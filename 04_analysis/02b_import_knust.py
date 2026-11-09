"""Step 02b: import the KNUST staff survey export and clean it with written rules.
Run:     python 04_analysis/02b_import_knust.py PATH_TO_EXPORT.csv
Input:   a CSV export from KoboToolbox or Google Forms, saved in 03_data/raw/ (never pushed).
Mapping: 02_instrument/knust_codebook.csv maps each survey column to an item name.
Output:  03_data/clean/knust_clean.csv and 03_data/clean/knust_screening.csv
Rule 0:  a row without all five consent boxes ticked is dropped first.
Never edit the raw export by hand. Every cleaning rule lives here.
"""
import sys

import numpy as np
import pandas as pd

import settings as S
from common import banner, done, leading_number

MIN_SECONDS_SHARE = 1 / 3                          # ASSUMPTION: a third of the pilot median time
# context questions (section A of the questionnaire), carried through unchanged when present
CONTEXT = ["age_band", "gender", "staff_category", "tenure_band", "training_12m", "sim_exposure", "lc_recall"]
# the five consent boxes on page 1 of the questionnaire (02_instrument/KNUST_Staff_Questionnaire.docx)
CONSENT = [f"consent{i}" for i in range(1, 6)]
NOT_TICKED = {"", "0", "0.0", "no", "false", "not ticked", "nan"}


def has_full_consent(raw):
    """True for rows where all five consent boxes are ticked. If the export has no consent
    columns (the online form made every box required), every row passes. A partial set is an error."""
    present = [c for c in CONSENT if c in raw.columns]
    if not present:
        return pd.Series(True, index=raw.index)
    if len(present) < len(CONSENT):
        raise SystemExit(f"Only {present} found. Export all five consent columns or none.")
    ticked = raw[CONSENT].apply(lambda col: ~col.astype(str).str.strip().str.lower().isin(NOT_TICKED) & col.notna())
    return ticked.all(axis=1)


def clean_knust(raw, codebook, pilot_median_seconds=None):
    cb = codebook.set_index("survey_column")
    missing = [c for c in cb.index if c not in raw.columns]
    if missing:
        raise SystemExit(f"These codebook columns are not in the export: {missing[:5]} ...\n"
                         "Fix: rename the survey questions to match 02_instrument/knust_codebook.csv")
    d = pd.DataFrame({cb.loc[c, "item"]: raw[c].map(leading_number) for c in cb.index})
    for c in cb.index:
        item, lo, hi = cb.loc[c, "item"], cb.loc[c, "min"], cb.loc[c, "max"]
        bad = d[item].notna() & ~d[item].between(lo, hi)
        if bad.any():
            raise SystemExit(f"{item}: {int(bad.sum())} value(s) outside {lo} to {hi}. Fix the export or the codebook.")
        if str(cb.loc[c, "reverse"]).lower() == "yes":
            d[item] = lo + hi - d[item]
    flags = pd.DataFrame(index=d.index)
    # rule 0: no full consent, no data. These rows are dropped before any other rule is applied.
    flags["no_consent"] = ~has_full_consent(raw).values
    # three checks. A check counts only if it was shown: section E (and att_lc) is skipped by staff
    # who cannot recall a phishing email. Anyone who answered no check at all is also flagged.
    att = codebook[codebook["scale"] == "attention"]
    wrong = [d[r.item].notna() & (d[r.item] != float(r.attention_key)) for r in att.itertuples()]
    none_seen = d[list(att["item"])].isna().all(axis=1)
    flags["failed_attention"] = (np.logical_or.reduce(wrong) | none_seen) & ~flags["no_consent"]
    if "duration_seconds" in raw.columns and pilot_median_seconds:
        flags["too_fast"] = raw["duration_seconds"] < MIN_SECONDS_SHARE * pilot_median_seconds
    else:
        flags["too_fast"] = False
    scale_items = [i for v in S.SCALE_ITEMS.values() for i in v if i in d]
    flags["straight_lined"] = d[scale_items].nunique(axis=1) == 1
    flags["duplicate"] = raw.duplicated(subset=[c for c in raw.columns if c in cb.index], keep="first")
    keep = ~flags.any(axis=1)
    for c in CONTEXT:
        if c in raw.columns:
            d[c] = raw[c].values
    out = d[keep].copy()
    out.insert(0, "site", "knust")
    out.insert(0, "id", np.arange(1, len(out) + 1))
    screen = flags.sum().rename("n_flagged").reset_index().rename(columns={"index": "rule"})
    screen.loc[len(screen)] = ["kept", int(keep.sum())]
    return out, screen


if __name__ == "__main__":
    banner("Step 02b", "import the KNUST survey", ["03_data/clean/knust_clean.csv", "knust_screening.csv"])
    if len(sys.argv) < 2:
        sys.exit("Usage: python 04_analysis/02b_import_knust.py 03_data/raw/YOUR_EXPORT.csv")
    raw = pd.read_csv(sys.argv[1])
    codebook = pd.read_csv(S.ROOT / "02_instrument" / "knust_codebook.csv")
    out, screen = clean_knust(raw, codebook)
    S.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(S.CLEAN_DIR / "knust_clean.csv", index=False)
    screen.to_csv(S.CLEAN_DIR / "knust_screening.csv", index=False)
    print(screen.to_string(index=False))
    done("Rscript 04_analysis/07_invariance.R")

"""Tests for parsing, scoring, sample rules and settings. Run: python -m pytest -q"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_analysis"))
import settings as S                     # noqa: E402
from common import leading_number, event, parse_timestamp   # noqa: E402

imp = importlib.import_module("02_fetch_import")
knust = importlib.import_module("02b_import_knust")


def test_leading_number_reads_likert_labels():
    assert leading_number("7. Strongly agree") == 7
    assert leading_number("1 (very low)") == 1
    assert leading_number("5 (extremly)") == 5
    assert np.isnan(leading_number(np.nan))


def test_event_codes_logs():
    s = pd.Series(["--", "24/02/2025 15:55:16", np.nan])
    out = event(s)
    assert out.iloc[0] == 0 and out.iloc[1] == 1 and np.isnan(out.iloc[2])


def test_timestamps_both_formats_and_swap():
    assert parse_timestamp("13/05/2025 10:43:29") == pd.Timestamp("2025-05-13 10:43:29")
    t = parse_timestamp("45996.5")             # Excel read 12/05/2025 as 5 Dec 2025
    assert (t.year, t.month, t.day) == (2025, 5, 12)
    assert pd.isna(parse_timestamp("--"))


def toy_raw(n=4):
    """A tiny raw table with every column the import uses."""
    raw = {"anonymisedEmail": ["a", "b", "c", "a"][:n], "Age": ["25-34"] * n, "Gender": ["Female", "Male", "Non-binary", "Female"][:n],
           "JobRole_std": ["academic", np.nan, "non-academic tech", "academic"][:n],
           "[anonymized institution]_years_groups": ["1-5"] * n, "cybsafeLinkClicked": [0] * n}
    for i in (1, 2):
        raw[f"Optimism_bias_self_{i}"] = ["2"] * n; raw[f"Optimism_bias_other_{i}"] = ["5"] * n
    for i in range(1, 5):
        raw[f"self_efficacy_{i}"] = ["6"] * n
    for i in range(1, 7):
        raw[f"risk_perception_{i}"] = ["6. Agree"] * n
    raw["risk_perception_5"] = ["1. Strongly disagree", np.nan, "4. Neither", "1. Strongly disagree"][:n]
    for i in range(1, 13):
        raw[f"locus_of_control_{i}"] = [2.0] * n
    for i in range(1, 9):
        raw[f"Acceptance _{i}"] = ["5. Somewhat agree"] * n
    raw["Acceptance _7"] = ["3. Somewhat disagree"] * n
    for i in range(1, 14):
        raw[f"negative_emotions_{i}"] = ["2 (a little)"] * n
    raw["negative_emotions_11"] = ["1 (very slightly or not at all)"] * n
    for i in range(1, 6):
        raw[f"Training intention _{i}"] = ["4. Neither"] * n
    for sim in ("sim1", "sim2"):
        for c in ("EmailLinkClicked_TimeStamp", "CredSupplied_TimeStamp_Compromised", "Phishing_Reported_On",
                  "MessageDeleted_TimeStamp"):
            raw[f"{sim}_{c}"] = ["--"] * n
        raw[f"{sim}_MessageRead_TimeStamp"] = ["01/02/2025 10:00:00"] * n
    raw["sim1_OutOfOfficeDays"] = [0] * n
    raw["sim1_EmailLinkClicked_TimeStamp"] = ["x", "--", "--", "x"][:n]
    for c in ("sim0_Viewed Landing Page_Time_c1_2023-03-29", "sim0.5_Viewed Landing Page_Time_c2_2024-07-22",
              "sim0_Data Entered_Time_c1_2023-03-29", "sim0.5_Data Entered_Time_c2_2024-07-22"):
        raw[c] = ["--", np.nan, "--", "--"][:n]
    return pd.DataFrame(raw)


def test_sample_rules_and_scoring():
    clean, flow = imp.build(toy_raw())
    assert len(clean) == 3                                 # duplicate "a" removed, first kept
    assert list(clean["rule_published"]) == [1, 0, 0]      # b never saw rp5, c failed it
    assert list(clean["rule_corrected"]) == [1, 1, 0]      # b kept: the item was not shown
    assert clean.loc[0, "lc1"] == 6                        # reversed: 8 - 2
    assert clean.loc[0, "optimism_bias"] == 3              # peer 5 - self 2
    assert clean.loc[1, "role"] == "missing" and clean.loc[1, "role_missing"] == 1
    assert clean.loc[2, "gender_other"] == 1               # kept in the causal models
    assert clean.loc[1, "history_missing"] == 1
    assert clean.loc[0, "caught3"] == 1 and clean.loc[1, "caught3"] == 0


def test_knust_import_rules():
    cb = pd.read_csv(S.ROOT / "02_instrument" / "knust_codebook.csv")
    row = {c: 4 for c in cb["survey_column"]}
    row.update({"rp5": 1, "att_ti": 3, "att_lc": 7, "se1": 5, "lc1": 2})
    bad = dict(row, att_ti=5)
    flat = {c: 4 for c in cb["survey_column"]}
    skipped = dict(row, **{f"lc{i}": np.nan for i in range(1, 13)}, att_lc=np.nan, se2=6)
    raw = pd.DataFrame([row, bad, flat, skipped])
    raw["staff_category"] = ["academic", "administrative", "technical", "academic"]
    out, screen = knust.clean_knust(raw, cb)
    assert len(out) == 2 and out["lc1"].iloc[0] == 6       # lc1 reversed on a 1-7 scale
    assert out["lc1"].isna().iloc[1]                       # skipped section E is kept, not failed
    assert out["staff_category"].iloc[0] == "academic"     # context answers carried through
    assert "consent1" not in out.columns                   # consent ticks never reach the analysis file


def test_knust_consent_rule():
    cb = pd.read_csv(S.ROOT / "02_instrument" / "knust_codebook.csv")
    row = {c: 4 for c in cb["survey_column"]}
    row.update({"rp5": 1, "att_ti": 3, "att_lc": 7, "se1": 5})
    raw = pd.DataFrame([row, row.copy(), row.copy(), row.copy()])
    raw["se2"] = [3, 5, 6, 2]                              # rows differ, so none is a duplicate
    for i in range(1, 6):
        raw[f"consent{i}"] = ["Yes", "1", "Yes", "Yes"]
    raw.loc[2, "consent5"] = ""                            # one box left empty
    raw.loc[3, "consent3"] = "No"                          # one box answered no
    out, screen = knust.clean_knust(raw, cb)
    assert len(out) == 2
    assert int(screen.set_index("rule").loc["no_consent", "n_flagged"]) == 2
    with pytest.raises(SystemExit):                        # a partial set of consent columns is refused
        knust.clean_knust(raw.drop(columns=["consent4"]), cb)


def test_no_post_treatment_controls():
    assert not set(S.COVARIATES) & set(S.MEDIATORS + ["training_intention"])


def test_settings_are_sane():
    assert S.DATA_MODE in ("rehearsal", "real")
    assert 0 < S.PS_CLIP < 0.1 and S.N_FOLDS >= 2

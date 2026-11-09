"""Step 02: fetch the open data, check it, and build the clean analysis table.
Run:     python 04_analysis/02_fetch_import.py              (downloads from Zenodo)
         python 04_analysis/02_fetch_import.py --zip PATH   (a zip you downloaded in a browser)
         python 04_analysis/02_fetch_import.py --xlsx PATH  (the data file itself, md5 not checked)
Output:  03_data/clean/staff_clean.csv and 03_data/clean/sample_flow.csv
The raw file stays in 03_data/raw, which is never pushed to GitHub (.gitignore).
"""
import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import settings as S
from common import leading_number, event

AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
TENURE = ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31+"]
RP_ITEMS = [1, 2, 3, 4, 6]            # item 5 is an attention check
AC_ITEMS = [1, 2, 3, 4, 5, 6, 8]      # item 7 is an attention check
NE_ITEMS = [1, 3, 4, 5, 6, 7, 9, 10, 12, 13]   # 2 and 8 are consistency words, 11 is a check
LC_REVERSED = [1, 2, 6, 7, 8, 9, 10, 11]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_raw(args):
    S.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.xlsx:
        print("Using the data file you gave. Its md5 cannot be checked against Zenodo.")
        return Path(args.xlsx)
    zpath = Path(args.zip) if args.zip else S.RAW_DIR / "zenodo_v4.zip"
    if not args.zip and not zpath.exists():
        print("Downloading from Zenodo (1.4 MB) ...")
        try:
            urllib.request.urlretrieve(S.ZENODO_URL, zpath)
        except Exception as e:                  # noqa: BLE001
            sys.exit(f"Download failed ({e}).\nFix: download the zip in a browser from "
                     "https://zenodo.org/records/23216625 and run again with --zip PATH")
    got = md5(zpath)
    if got != S.ZENODO_ZIP_MD5:
        sys.exit(f"md5 mismatch: {got}\nFix: download the zip again. Do not use an edited copy.")
    print("md5 check passed:", got)
    with zipfile.ZipFile(zpath) as z:
        member = [m for m in z.namelist() if m.endswith(S.DATA_FILE_IN_ZIP)]
        if not member:
            sys.exit("The zip has no study_data_merged.xlsx. Fix: check you downloaded version 4.")
        out = S.RAW_DIR / S.DATA_FILE_IN_ZIP
        out.write_bytes(z.read(member[0]))
    return out


def build(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = raw.copy()
    n_recorded = len(d)
    d = d.drop_duplicates("anonymisedEmail", keep="first").reset_index(drop=True)   # first entry kept
    n_dedup = len(d)

    # numeric items
    def items(prefix, idx):
        return pd.DataFrame({f"{prefix}{i}": d[col].map(leading_number) for i, col in idx})
    ob = items("ob_", [("self1", "Optimism_bias_self_1"), ("self2", "Optimism_bias_self_2"),
                       ("peer1", "Optimism_bias_other_1"), ("peer2", "Optimism_bias_other_2")])
    se = items("se", [(i, f"self_efficacy_{i}") for i in range(1, 5)])
    rp = items("rp", [(i, f"risk_perception_{i}") for i in range(1, 7)])
    lc = items("lc", [(i, f"locus_of_control_{i}") for i in range(1, 13)])
    for i in LC_REVERSED:
        lc[f"lc{i}"] = 8 - lc[f"lc{i}"]
    ac = items("ac", [(i, f"Acceptance _{i}") for i in range(1, 9)])
    ne = items("ne", [(i, f"negative_emotions_{i}") for i in range(1, 14)])
    ti = items("ti", [(i, f"Training intention _{i}") for i in range(1, 6)])

    # attention rules
    pass_rp, pass_ac, pass_ne = rp["rp5"] == 1, ac["ac7"] == 3, ne["ne11"] == 1
    rule_pub = pass_rp & pass_ac & pass_ne
    rule_cor = (rp["rp5"].isna() | pass_rp) & pass_ac & pass_ne      # only items the version showed

    # exposure to Sims 3 and 4 (read or deleted)
    def seen(c):
        return event(d[c]).fillna(0) == 1
    exposed = (seen("sim1_MessageRead_TimeStamp") | seen("sim1_MessageDeleted_TimeStamp")) & \
              (seen("sim2_MessageRead_TimeStamp") | seen("sim2_MessageDeleted_TimeStamp"))

    out = pd.DataFrame({"id": np.arange(1, n_dedup + 1)})
    out["rule_published"] = (rule_pub & exposed).astype(int)
    out["rule_corrected"] = (rule_cor & exposed).astype(int)

    # demographics
    out["age_num"] = d["Age"].map({k: i + 1 for i, k in enumerate(AGE_BANDS)})
    out["age_band"] = out["age_num"]
    out["age3"] = pd.cut(out["age_num"], [0, 2, 4, 6], labels=["<=34", "35-49", "50+"]).astype(str)
    g = d["Gender"].fillna("Prefer not to answer")
    out["gender"] = g
    out["gender_bin"] = g.map({"Female": 1, "Male": 0})                  # baseline coding (others NaN)
    out["female"] = (g == "Female").astype(int)
    out["gender_other"] = (~g.isin(["Female", "Male"])).astype(int)       # keeps everyone in the model
    role = d["JobRole_std"].fillna("missing")
    out["role"] = role
    out["jobrole_bin01"] = np.where(role == "missing", np.nan, (role != "academic").astype(float))
    out["role_nonacad"] = (role == "non-academic").astype(int)
    out["role_tech"] = (role == "non-academic tech").astype(int)
    out["role_missing"] = (role == "missing").astype(int)
    ten = d["[anonymized institution]_years_groups"].astype(str).str.strip()
    out["years_org_num"] = ten.map({k: i + 1 for i, k in enumerate(TENURE)})
    out["tenure"] = out["years_org_num"]

    # behaviour: Sims 1-2 (CybSafe) and 3-4 (Defender)
    out["click1"] = event(d["sim0_Viewed Landing Page_Time_c1_2023-03-29"])
    out["click2"] = event(d["sim0.5_Viewed Landing Page_Time_c2_2024-07-22"])
    out["comp1"] = event(d["sim0_Data Entered_Time_c1_2023-03-29"])
    out["comp2"] = event(d["sim0.5_Data Entered_Time_c2_2024-07-22"])
    out["history_missing"] = out["click1"].isna().astype(int)
    out["prior_click"] = ((out["click1"] == 1) | (out["click2"] == 1)).astype(int)
    out["prior_comp"] = ((out["comp1"] == 1) | (out["comp2"] == 1)).astype(int)
    out["prior"] = np.select([out["history_missing"] == 1, out["prior_click"] == 1],
                             ["history unknown", "prior click"], "no prior click")
    out["click3"] = event(d["sim1_EmailLinkClicked_TimeStamp"])
    out["comp3"] = event(d["sim1_CredSupplied_TimeStamp_Compromised"])
    out["click4"] = event(d["sim2_EmailLinkClicked_TimeStamp"])
    out["comp4"] = event(d["sim2_CredSupplied_TimeStamp_Compromised"])
    out["report3"] = event(d["sim1_Phishing_Reported_On"])
    out["report4"] = event(d["sim2_Phishing_Reported_On"])
    out["caught3"] = out["click3"]
    out["lvl3"] = np.select([out["comp3"] == 1, out["click3"] == 1], [2, 1], 0)
    out["lvl4"] = np.select([out["comp4"] == 1, out["click4"] == 1], [2, 1], 0)
    out["ooo_days3"] = pd.to_numeric(d["sim1_OutOfOfficeDays"], errors="coerce").fillna(0)
    out["training_link"] = d["cybsafeLinkClicked"]

    # items and composites (baseline scoring)
    out = pd.concat([out, ob, se, rp, lc, ac, ne, ti], axis=1)
    out["optimism_bias"] = ob[["ob_peer1", "ob_peer2"]].mean(axis=1) - ob[["ob_self1", "ob_self2"]].mean(axis=1)
    out["self_efficacy"] = se.mean(axis=1)
    out["risk_perception"] = rp[[f"rp{i}" for i in RP_ITEMS]].mean(axis=1)
    out["locus_of_control"] = lc.mean(axis=1)
    out["acceptance"] = ac[[f"ac{i}" for i in AC_ITEMS]].mean(axis=1)
    out["negative_emotions"] = ne[[f"ne{i}" for i in NE_ITEMS]].mean(axis=1)
    out["training_intention"] = ti.mean(axis=1)
    out["site"] = "open_data"

    flow = pd.DataFrame({
        "step": ["recorded", "after duplicates removed", "published rule (all checks + exposure)",
                 "corrected rule (checks shown + exposure)"],
        "n": [n_recorded, n_dedup, int(out["rule_published"].sum()), int(out["rule_corrected"].sum())]})
    return out, flow


if __name__ == "__main__":
    from common import banner, done
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip"); ap.add_argument("--xlsx")
    args = ap.parse_args()
    banner("Step 02", "fetch and import the open data", ["03_data/clean/staff_clean.csv", "sample_flow.csv"])
    raw_path = get_raw(args)
    clean, flow = build(pd.read_excel(raw_path))
    S.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(S.CLEAN_DIR / "staff_clean.csv", index=False)
    flow.to_csv(S.CLEAN_DIR / "sample_flow.csv", index=False)
    print(flow.to_string(index=False))
    for rule, n in S.EXPECTED_N.items():
        got = int(clean[f"rule_{rule}"].sum())
        print(f"  {rule:9s} rule: N = {got} (expected {n}) {'OK' if got == n else 'CHECK: differs'}")
    done("set DATA_MODE = 'real' in settings.py only after the preregistration is posted, then run step 03")

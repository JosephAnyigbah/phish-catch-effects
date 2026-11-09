"""Every analysis setting lives here. No script hard-codes a setting.

Change DATA_MODE to "real" only after:
  1. 02_fetch_import.py has built 03_data/clean/staff_clean.csv, and
  2. the preregistration (06_docs/PREREGISTRATION.md) is posted on OSF.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- mode -------------------------------------------------------------
DATA_MODE = "rehearsal"            # "rehearsal" (demo) or "real"
SEED = 2027

# ---- paths ------------------------------------------------------------
RAW_DIR = ROOT / "03_data" / "raw"
REHEARSAL_DIR = ROOT / "03_data" / "rehearsal"
CLEAN_DIR = ROOT / "03_data" / "clean"
TABLES = ROOT / "05_results" / "tables"
FIGURES = ROOT / "05_results" / "figures"

# ---- open data (George et al., Zenodo v4) -----------------------------
ZENODO_URL = ("https://zenodo.org/records/23216625/files/"
              "increased-and-repeated-clicking-behavior-analysis-camera-ready_8cfdb0b.zip?download=1")
ZENODO_ZIP_MD5 = "3cb0e6f97e5cfac0906e0c3cd29029d5"
DATA_FILE_IN_ZIP = "study_data_merged.xlsx"

# ---- sample rules -----------------------------------------------------
# "published": all three attention items must match (paper: N = 986)
# "corrected": an item counts only if the respondent's version showed it (N = 1,269)
SAMPLE_RULE_BENCHMARK = "published"
SAMPLE_RULE_CAUSAL = "corrected"
EXPECTED_N = {"published": 986, "corrected": 1269}

# ---- causal arm (double machine learning) -----------------------------
N_FOLDS = 5
N_REPEATS = 10                      # use 3 in the demo for speed
N_REPEATS_DEMO = 3
PS_CLIP = 0.02                      # propensity scores clipped to [0.02, 0.98]
TREATMENT = "caught3"               # clicked in Simulation 3
OUTCOMES = ["click4", "comp4", "report4"]
# pre-treatment covariates only: the six survey scales were measured AFTER Sim 3
COVARIATES = ["age_band", "female", "gender_other", "role_nonacad", "role_tech",
              "role_missing", "tenure", "prior_click", "prior_comp", "history_missing",
              "ooo_days3"]
GROUPS = {
    "role": ["academic", "non-academic", "non-academic tech"],
    "age3": ["<=34", "35-49", "50+"],
    "prior": ["no prior click", "prior click", "history unknown"],
}
MIN_CELL = 10                       # never publish a table cell below this

# ---- benchmark path model ---------------------------------------------
MEDIATORS = ["optimism_bias", "risk_perception", "negative_emotions",
             "acceptance", "self_efficacy", "locus_of_control"]
CONTROLS_BENCHMARK = ["age_num", "gender_bin", "jobrole_bin01", "years_org_num"]

# baseline values used to judge the rebuild (REPORTED, Tab 5 and Fig 3)
BASELINE_H2 = {"chi2": 208.46, "df": 15,
               "ind_B": {"optimism_bias": -0.010, "risk_perception": -0.002,
                         "negative_emotions": 0.116, "acceptance": 0.038,
                         "self_efficacy": -0.037, "locus_of_control": 0.018},
               "ind_SE": {"optimism_bias": 0.009, "risk_perception": 0.009,
                          "negative_emotions": 0.026, "acceptance": 0.015,
                          "self_efficacy": 0.017, "locus_of_control": 0.012},
               "a_std": {"optimism_bias": -0.056, "risk_perception": -0.006,
                         "negative_emotions": 0.217, "acceptance": 0.100,
                         "self_efficacy": -0.160, "locus_of_control": -0.068},
               "b_std": {"optimism_bias": 0.073, "risk_perception": 0.124,
                         "negative_emotions": 0.217, "acceptance": 0.154,
                         "self_efficacy": 0.094, "locus_of_control": -0.110},
               "total_indirect": (0.124, 0.039), "total": (0.062, 0.077)}
BASELINE_H1_TOTAL = (0.086, 0.043)  # Tab 4: B and SE
CLOSE_ENOUGH = 0.005                # |rebuilt - reported| for "replicated"

# ---- scales tested at KNUST (items as named in the clean table) -------
SCALE_ITEMS = {
    "optimism_self": ["ob_self1", "ob_self2"],
    "optimism_peer": ["ob_peer1", "ob_peer2"],
    "self_efficacy": ["se1", "se2", "se3", "se4"],
    "risk_perception": ["rp1", "rp2", "rp3", "rp4", "rp6"],     # rp5 is an attention check
    "locus_of_control": [f"lc{i}" for i in range(1, 13)],
    "training_intention": ["ti1", "ti2", "ti3", "ti4", "ti5"],
}

# ---- figures ----------------------------------------------------------
PALETTE = ["#1F3864", "#E69F00", "#56B4E9", "#009E73"]   # colour-blind safe
WATERMARK = "SYNTHETIC DEMO DATA - not a result"

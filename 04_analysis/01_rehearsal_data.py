"""Step 01: build REHEARSAL data from the baseline's reported numbers. Never report these numbers.
Run:     python 04_analysis/01_rehearsal_data.py
Output:  03_data/rehearsal/REHEARSAL_staff.csv   (same columns as the real clean table)
         03_data/rehearsal/REHEARSAL_knust.csv   (a pretend KNUST survey for the invariance demo)
         03_data/rehearsal/REHEARSAL_truth.csv   (the effects planted in the data)
Sources of the numbers: sample shape and rates (paper 3.2.1, 4.1), a- and b-paths (Fig 3),
scale correlations (Tab 3). Shapes not reported are ASSUMPTION and marked below.
"""
import numpy as np
import pandas as pd

import settings as S
from common import banner, done

rng = np.random.default_rng(S.SEED)
N = 1269                                   # corrected-rule size; 986 of them pass the published rule

# ---------------- demographics (rates from the paper, 3.2.1) ----------------
age_num = rng.choice(np.arange(1, 7), N, p=[.04, .34, .30, .21, .09, .02])        # ASSUMPTION shape
gender = rng.choice(["Female", "Male", "Non-binary", "Prefer not to answer"], N, p=[.588, .371, .014, .027])
role = rng.choice(["academic", "non-academic", "non-academic tech", "missing"], N, p=[.461, .445, .082, .012])
tenure = rng.choice(np.arange(1, 8), N, p=[.40, .22, .14, .10, .07, .04, .03])    # ASSUMPTION shape
history_missing = rng.binomial(1, 0.30, N)                                         # 692 of 986 had history
prior_click = np.where(history_missing == 1, 0, rng.binomial(1, 0.20, N))
prior_comp = prior_click * rng.binomial(1, 0.25, N)
# Sims 1 and 2 separately (NaN where the history is unknown); prior_click = clicked in either
click1 = np.where(prior_click == 1, rng.binomial(1, 0.25, N), 0)
click2 = np.where((prior_click == 1) & (click1 == 0), 1, np.where(prior_click == 1, rng.binomial(1, 0.7, N), 0))
comp1 = click1 * rng.binomial(1, 0.1, N)
click1, click2, comp1 = (np.where(history_missing == 1, np.nan, v).astype(float) for v in (click1, click2, comp1))
ooo_days3 = rng.poisson(0.3, N)

logit = lambda z: 1 / (1 + np.exp(-z))
# ---------------- Simulation 3: being caught (about 18% click, 6% compromise) ----------------
p_caught = logit(-1.75 + 0.9 * prior_click + 0.3 * (role == "non-academic tech") - 0.08 * (age_num - 3))
caught3 = rng.binomial(1, p_caught)
comp3 = caught3 * rng.binomial(1, 0.33, N)
lvl3 = np.where(comp3 == 1, 2, np.where(caught3 == 1, 1, 0))

# ---------------- Simulation 4 with PLANTED effects (rehearsal truth) ----------------
TRUE_CLICK = {"no prior click": 0.06, "prior click": 0.16, "history unknown": 0.06}
TRUE_REPORT = -0.05
prior_lab = np.where(history_missing == 1, "history unknown", np.where(prior_click == 1, "prior click", "no prior click"))
eff_click = np.vectorize(TRUE_CLICK.get)(prior_lab)
base4 = logit(-0.85 + 0.8 * prior_click + 0.1 * (role == "non-academic tech"))
click4 = rng.binomial(1, np.clip(base4 + eff_click * caught3, 0, 1))
comp4 = click4 * rng.binomial(1, 0.27, N)
report4 = rng.binomial(1, np.clip(0.35 - 0.10 * click4 + TRUE_REPORT * caught3, 0, 1))
report3 = rng.binomial(1, 0.33 * (1 - caught3) + 0.10 * caught3)

# ---------------- survey scales: a-paths and b-paths from Fig 3 (standardised) ----------------
names = S.MEDIATORS
a_std = np.array([-0.056, -0.006, 0.217, 0.100, -0.160, -0.068])
b_std = np.array([0.073, 0.124, 0.217, 0.154, 0.094, -0.110])
R = np.array([[1, -.01, -.10, .02, .33, .23],      # Tab 3 order mapped to MEDIATORS order (REPORTED)
              [-.01, 1, .15, .24, -.02, .01],
              [-.10, .15, 1, .06, -.20, -.14],
              [.02, .24, .06, 1, -.05, .13],
              [.33, -.02, -.20, -.05, 1, .31],
              [.23, .01, -.14, .13, .31, 1]])
z3 = (lvl3 - lvl3.mean()) / lvl3.std()
resid = rng.multivariate_normal(np.zeros(6), R * (1 - (a_std ** 2).mean()), N)
M = z3[:, None] * a_std + resid
TI = M @ b_std - 0.025 * z3 + rng.normal(0, 0.9, N)

def likert(latent, k, lo, hi, loading=0.8, shift=0.0):
    """k Likert items from one latent score (ASSUMPTION: loadings about 0.8)."""
    items = loading * latent[:, None] + rng.normal(0, np.sqrt(1 - loading ** 2), (len(latent), k))
    mid, spread = (lo + hi) / 2, (hi - lo) / 4
    return np.clip(np.rint(mid + shift + spread * items), lo, hi)

def item_name(key, j):
    """Real column names: risk perception skips rp5 (an attention check in the real survey)."""
    if key.startswith("ob_"):
        return f"ob_{key.split('_')[1]}{j + 1}"
    if key == "rp":
        return f"rp{[1, 2, 3, 4, 6][j]}"
    return f"{key}{j + 1}"


df = pd.DataFrame({"id": np.arange(1, N + 1)})
pub = np.zeros(N, int); pub[rng.choice(N, 986, replace=False)] = 1
df["rule_published"], df["rule_corrected"] = pub, 1
df["age_num"] = age_num; df["age_band"] = age_num
df["age3"] = pd.cut(age_num, [0, 2, 4, 6], labels=["<=34", "35-49", "50+"]).astype(str)
df["gender"] = gender
df["gender_bin"] = pd.Series(gender).map({"Female": 1, "Male": 0})
df["female"] = (gender == "Female").astype(int)
df["gender_other"] = (~np.isin(gender, ["Female", "Male"])).astype(int)
df["role"] = role
df["jobrole_bin01"] = np.where(role == "missing", np.nan, (role != "academic").astype(float))
df["role_nonacad"] = (role == "non-academic").astype(int)
df["role_tech"] = (role == "non-academic tech").astype(int)
df["role_missing"] = (role == "missing").astype(int)
df["years_org_num"] = tenure; df["tenure"] = tenure
df["history_missing"] = history_missing; df["prior_click"] = prior_click; df["prior_comp"] = prior_comp
df["prior"] = prior_lab
df["click1"], df["click2"], df["comp1"] = click1, click2, comp1
for c, v in {"caught3": caught3, "click3": caught3, "comp3": comp3, "click4": click4, "comp4": comp4,
             "report3": report3, "report4": report4, "lvl3": lvl3,
             "lvl4": np.where(comp4 == 1, 2, np.where(click4 == 1, 1, 0)), "ooo_days3": ooo_days3}.items():
    df[c] = v
df["training_link"] = rng.binomial(1, logit(-2.9 + 0.25 * TI))

# items, then composites scored as in the paper
# self and peer ratings share only part of the optimism signal (ASSUMPTION), as in real data
ob_self = likert(-0.6 * M[:, 0] + 0.8 * rng.normal(0, 1, N), 2, 1, 7, shift=-0.5)
ob_peer = likert(0.6 * M[:, 0] + 0.8 * rng.normal(0, 1, N), 2, 1, 7, shift=0.5)
blocks = {"ob_self": ob_self, "ob_peer": ob_peer, "se": likert(M[:, 4], 4, 1, 7, shift=1.5),
          "rp": likert(M[:, 1], 5, 1, 7, shift=2.3), "lc": likert(M[:, 5], 12, 1, 7, loading=0.6),
          "ac": likert(M[:, 3], 7, 1, 7, shift=1.3), "ne": likert(M[:, 2], 10, 1, 5, shift=-1.4),
          "ti": likert(TI / TI.std(), 5, 1, 7, shift=-0.2)}
for key, arr in blocks.items():
    for j in range(arr.shape[1]):
        df[item_name(key, j)] = arr[:, j]
# planned missingness: each respondent skipped one of OB, SE, LC (C11 pattern)
skip = rng.choice(["ob", "se", "lc"], N)
for blk, cols in {"ob": ["ob_self1", "ob_self2", "ob_peer1", "ob_peer2"], "se": [f"se{i}" for i in range(1, 5)],
                  "lc": [f"lc{i}" for i in range(1, 13)]}.items():
    df.loc[skip == blk, cols] = np.nan
df["optimism_bias"] = df[["ob_peer1", "ob_peer2"]].mean(axis=1) - df[["ob_self1", "ob_self2"]].mean(axis=1)
df["self_efficacy"] = df[[f"se{i}" for i in range(1, 5)]].mean(axis=1)
df["risk_perception"] = df[S.SCALE_ITEMS["risk_perception"]].mean(axis=1)
df["locus_of_control"] = df[[f"lc{i}" for i in range(1, 13)]].mean(axis=1)
df["acceptance"] = df[[f"ac{i}" for i in range(1, 8)]].mean(axis=1)
df["negative_emotions"] = df[[f"ne{i}" for i in range(1, 11)]].mean(axis=1)
df["training_intention"] = df[[f"ti{i}" for i in range(1, 6)]].mean(axis=1)
df["site"] = "open_data"

# pretend KNUST survey: five scales, all items, one intercept shifted (partial scalar demo)
K = 250
Lk = rng.multivariate_normal(np.zeros(6), R, K)
kn = pd.DataFrame({"id": np.arange(1, K + 1), "site": "knust"})
for key, arr in {"ob_self": likert(-0.6 * Lk[:, 0] + 0.8 * rng.normal(0, 1, K), 2, 1, 7, shift=-0.5),
                 "ob_peer": likert(0.6 * Lk[:, 0] + 0.8 * rng.normal(0, 1, K), 2, 1, 7, shift=0.5),
                 "se": likert(Lk[:, 4], 4, 1, 7, shift=1.2), "rp": likert(Lk[:, 1], 5, 1, 7, shift=2.3),
                 "lc": likert(Lk[:, 5], 12, 1, 7, loading=0.6),
                 "ti": likert(rng.normal(0, 1, K), 5, 1, 7, shift=0.3)}.items():
    for j in range(arr.shape[1]):
        kn[item_name(key, j)] = arr[:, j]
kn["se3"] = np.clip(kn["se3"] - 1, 1, 7)          # planted non-invariant intercept (rehearsal truth)

truth = pd.DataFrame({"quantity": ["click effect, no prior click", "click effect, prior click",
                                   "click effect, history unknown", "report effect (direct -0.05, plus about -0.01 through extra clicking)",
                                   "KNUST non-invariant item"],
                      "planted_value": [0.06, 0.16, 0.06, -0.06, "se3 intercept"]})

if __name__ == "__main__":
    banner("Step 01", "build rehearsal data", ["REHEARSAL_staff.csv", "REHEARSAL_knust.csv", "REHEARSAL_truth.csv"])
    S.REHEARSAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(S.REHEARSAL_DIR / "REHEARSAL_staff.csv", index=False)
    kn.to_csv(S.REHEARSAL_DIR / "REHEARSAL_knust.csv", index=False)
    truth.to_csv(S.REHEARSAL_DIR / "REHEARSAL_truth.csv", index=False)
    print(f"  {N} rehearsal staff ({pub.sum()} pass the published rule), {K} rehearsal KNUST staff")
    print("  caught in Sim 3: {:.1%}   clicked Sim 4: {:.1%}   reported Sim 4: {:.1%}".format(
        caught3.mean(), click4.mean(), report4.mean()))
    done("python 04_analysis/03_screening.py")

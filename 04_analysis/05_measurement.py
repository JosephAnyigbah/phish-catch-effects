"""Step 05: measurement model for the five scales tested at KNUST (CFA per scale, reliability, HTMT).
Run:     python 04_analysis/05_measurement.py
Output:  05_results/tables/T6_measurement.csv, T6b_htmt.csv
Each CFA uses the respondents whose questionnaire version showed that scale (planned missingness).
Locus of control is fitted twice: as one factor (the paper's composite) and as the four CDSII subscales.
"""
import warnings

import numpy as np
import pandas as pd
import semopy

import settings as S
from common import banner, done, load_staff, save_table

warnings.filterwarnings("ignore")
HTMT_BOOT = 50 if S.DATA_MODE == "rehearsal" else 500
CDSII = {"locus": ["lc1", "lc6", "lc9"], "external": ["lc5", "lc8", "lc12"],
         "stability": ["lc3", "lc7", "lc11"], "personal": ["lc2", "lc4", "lc10"]}


def cfa(data, factors):
    desc = "\n".join(f"{f} =~ " + " + ".join(items) for f, items in factors.items())
    items = [i for v in factors.values() for i in v]
    x = data[items].dropna()
    z = (x - x.mean()) / x.std()
    m = semopy.Model(desc)
    m.fit(z)
    est = m.inspect(std_est=True)
    st = semopy.calc_stats(m)
    rows = []
    for f, its in factors.items():
        lam = est[(est["op"] == "~") & (est["rval"] == f) & (est["lval"].isin(its))]["Est. Std"].to_numpy()
        if len(lam) == 0:             # semopy lists loadings as item ~ factor
            lam = est[(est["op"] == "~") & (est["lval"].isin(its)) & (est["rval"] == f)]["Est. Std"].to_numpy()
        cr = lam.sum() ** 2 / (lam.sum() ** 2 + (1 - lam ** 2).sum())
        xi = x[its]
        k = len(its)
        alpha = k / (k - 1) * (1 - xi.var().sum() / xi.sum(axis=1).var())
        rows.append({"factor": f, "n": len(x), "items": k, "alpha": round(alpha, 3),
                     "loading_min": round(lam.min(), 3), "loading_max": round(lam.max(), 3),
                     "CR": round(cr, 3), "AVE": round((lam ** 2).mean(), 3),
                     "CFI": round(float(st["CFI"].iloc[0]), 3), "RMSEA": round(float(st["RMSEA"].iloc[0]), 3)})
    return rows


def htmt(data, a, b):
    """Heterotrait-monotrait ratio. Items within a scale point the same way, so signed
    correlations are averaged and the absolute value is taken at the end."""
    r = data[a + b].corr()
    het = abs(r.loc[a, b].to_numpy().mean())
    mono = lambda its: r.loc[its, its].to_numpy()[np.triu_indices(len(its), 1)].mean()
    return het / np.sqrt(mono(a) * mono(b))


banner("Step 05", "measurement model", ["T6_measurement", "T6b_htmt"])
d = load_staff()
s = d[d[f"rule_{S.SAMPLE_RULE_BENCHMARK}"] == 1]
I = S.SCALE_ITEMS
rows = []
rows += cfa(s, {"optimism_self": I["optimism_self"], "optimism_peer": I["optimism_peer"]})
for name in ["self_efficacy", "risk_perception", "locus_of_control", "training_intention"]:
    rows += cfa(s, {name: I[name]})
rows += cfa(s, CDSII)
t6 = pd.DataFrame(rows)
print(t6.to_string(index=False))
print("Conventions (VERIFY sources): loadings 0.708, CR 0.70, AVE 0.50, CFI 0.95, RMSEA 0.08.")
save_table(t6, "T6_measurement")

# HTMT with bootstrap 95% CIs (pairwise complete data)
names = ["self_efficacy", "risk_perception", "locus_of_control", "training_intention"]
rng = np.random.default_rng(S.SEED)
out = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = I[names[i]], I[names[j]]
        x = s[a + b].dropna()
        if len(x) < 30:
            continue
        est = htmt(x, a, b)
        boots = [htmt(x.iloc[rng.integers(0, len(x), len(x))], a, b) for _ in range(HTMT_BOOT)]
        out.append({"pair": f"{names[i]} / {names[j]}", "n": len(x), "HTMT": round(est, 3),
                    "ci_low": round(np.percentile(boots, 2.5), 3), "ci_high": round(np.percentile(boots, 97.5), 3)})
t6b = pd.DataFrame(out)
print("\n" + (t6b.to_string(index=False) if len(t6b) else "No scale pair shares 30 complete respondents."))
print(f"(bootstrap samples: {HTMT_BOOT}; threshold 0.85 or 0.90 is a convention)")
save_table(t6b, "T6b_htmt")
done("python 04_analysis/06_causal_dml.py")

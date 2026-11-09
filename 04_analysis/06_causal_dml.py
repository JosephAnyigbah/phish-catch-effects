"""Step 06: effect of being caught in Simulation 3 on Simulation 4 behaviour (primary objective).
Run:     python 04_analysis/06_causal_dml.py
Output:  T2_ate.csv (average effects), T3_gates.csv (group effects), T3b_heterogeneity.csv,
         T3c_page_contrast.csv, T3d_sensitivity.csv, T3e_refuters.csv, T3f_forest_cate.csv
Rules:   adjust ONLY for variables measured before Simulation 3 (settings.COVARIATES).
         On real data, run this only after the preregistration is posted on OSF.
"""
import warnings

import numpy as np
import pandas as pd
import doubleml as dml
from doubleml.utils import PSProcessorConfig
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import settings as S
from common import banner, done, load_staff, save_table

warnings.filterwarnings("ignore")
np.random.seed(S.SEED)                        # DoubleML sample splits use the global seed
DEMO = S.DATA_MODE == "rehearsal"
REPS = S.N_REPEATS_DEMO if DEMO else S.N_REPEATS
LEARNERS = {
    "logit": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "boosting": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                       random_state=S.SEED),
}
PRIMARY = "logit"


def irm(df, y, d, x, learner):
    data = dml.DoubleMLData(df, y_col=y, d_cols=d, x_cols=x)
    m = dml.DoubleMLIRM(data, ml_g=LEARNERS[learner](), ml_m=LEARNERS[learner](), n_folds=S.N_FOLDS,
                        n_rep=REPS, ps_processor_config=PSProcessorConfig(clipping_threshold=S.PS_CLIP))
    m.fit()
    return m


def evalue(rr):
    rr = rr if rr >= 1 else 1 / rr
    return rr + np.sqrt(rr * (rr - 1))


banner("Step 06", "causal arm (double machine learning)",
       ["T2_ate", "T3_gates", "T3b_heterogeneity", "T3c_page_contrast", "T3d_sensitivity", "T3e_refuters", "T3f_forest_cate"])
d = load_staff()
s = d[d[f"rule_{S.SAMPLE_RULE_CAUSAL}"] == 1].dropna(subset=[S.TREATMENT] + S.OUTCOMES).reset_index(drop=True)
X = S.COVARIATES
print(f"N = {len(s)} ({S.SAMPLE_RULE_CAUSAL} rule). Treated (caught in Sim 3): {int(s[S.TREATMENT].sum())}. "
      f"Repeats: {REPS}.")

# ---- 1. average effects, two learner sets ----
rows, fitted = [], {}
for y in S.OUTCOMES:
    for name in LEARNERS:
        m = irm(s, y, S.TREATMENT, X, name)
        fitted[(y, name)] = m
        ci = m.confint().iloc[0]
        p0 = float(np.mean(m.predictions["ml_g0"]))
        rr = (p0 + m.coef[0]) / p0
        rows.append({"outcome": y, "learner": name, "ATE": round(m.coef[0], 4), "SE": round(m.se[0], 4),
                     "ci_low": round(ci.iloc[0], 4), "ci_high": round(ci.iloc[1], 4), "p": round(m.pval[0], 4),
                     "risk_untreated": round(p0, 3), "E_value": round(evalue(rr), 2)})
t2 = pd.DataFrame(rows)
print("\nAverage effects (risk differences):\n" + t2.to_string(index=False))
save_table(t2, "T2_ate")

# ---- 2. group effects (GATES) and a heterogeneity test, click outcome, primary learner ----
m = fitted[("click4", PRIMARY)]
grows, hrows = [], []
for gname, levels in S.GROUPS.items():
    groups = pd.DataFrame({lv: s[gname] == lv for lv in levels if (s[gname] == lv).any()})
    gate = m.gate(groups=groups)
    ci = gate.confint()
    for lv in groups.columns:
        n_treat = int(s.loc[groups[lv], S.TREATMENT].sum())
        grows.append({"grouping": gname, "group": lv, "n": int(groups[lv].sum()), "n_treated": n_treat,
                      "effect": round(ci.loc[lv, "effect"], 4), "ci_low": round(ci.loc[lv, "2.5 %"], 4),
                      "ci_high": round(ci.loc[lv, "97.5 %"], 4),
                      "publishable": "yes" if n_treat >= S.MIN_CELL else f"no: fewer than {S.MIN_CELL} treated"})
    ols = gate.blp_model[0] if isinstance(gate.blp_model, list) else gate.blp_model
    k = len(groups.columns)
    if k > 1:
        R = np.zeros((k - 1, k))
        for i in range(k - 1):
            R[i, 0], R[i, i + 1] = 1, -1
        f = ols.f_test(R)
        hrows.append({"grouping": gname, "test": "all group effects equal", "F": round(float(f.fvalue), 3),
                      "df": f"{int(f.df_num)}, {int(f.df_denom)}", "p": round(float(f.pvalue), 4)})
t3 = pd.DataFrame(grows)
print("\nGroup effects on Sim 4 clicking:\n" + t3.to_string(index=False))
save_table(t3, "T3_gates")
t3b = pd.DataFrame(hrows)
print("\nHeterogeneity tests (first repetition of the sample split):\n" + t3b.to_string(index=False))
save_table(t3b, "T3b_heterogeneity")

# ---- 3. training page against no page, among Sim 3 clickers ----
c = s[s["click3"] == 1].reset_index(drop=True)
pc = irm(c, "click4", "comp3", X, PRIMARY)
ci = pc.confint().iloc[0]
t3c = pd.DataFrame([{"contrast": "page shown (compromised) vs click only", "n": len(c), "n_page": int(c["comp3"].sum()),
                     "effect_on_click4": round(pc.coef[0], 4), "ci_low": round(ci.iloc[0], 4), "ci_high": round(ci.iloc[1], 4),
                     "p": round(pc.pval[0], 4)}])
print("\nTraining-page contrast:\n" + t3c.to_string(index=False))
save_table(t3c, "T3c_page_contrast")

# ---- 4. sensitivity to hidden confounding, and a negative-control outcome ----
srows = []
for y in ["click4", "report4"]:
    mm = fitted[(y, PRIMARY)]
    mm.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0)
    sp = mm.sensitivity_params
    srows.append({"check": f"{y}: bounds if hidden confounding is as strong as cf_y = cf_d = 0.03",
                  "lower": round(float(np.ravel(sp["theta"]["lower"])[0]), 4),
                  "upper": round(float(np.ravel(sp["theta"]["upper"])[0]), 4),
                  "robustness_value": round(float(np.ravel(sp["rv"])[0]), 4)})
h = s[s["history_missing"] == 0].reset_index(drop=True)
if all(col in h.columns for col in ["click1", "click2", "comp1"]):
    xs = [v for v in X if v not in ("prior_click", "prior_comp", "history_missing")] + ["click1", "comp1"]
    nc = irm(h.dropna(subset=["click2", "click1", "comp1"]), "click2", S.TREATMENT, xs, PRIMARY)
    ci = nc.confint().iloc[0]
    srows.append({"check": "negative control: Sim 3 'effect' on Sim 2 clicking (should be near 0)",
                  "lower": round(ci.iloc[0], 4), "upper": round(ci.iloc[1], 4), "robustness_value": np.nan})
else:
    srows.append({"check": "negative control: not run (Sim 1 and Sim 2 columns absent in this table)",
                  "lower": np.nan, "upper": np.nan, "robustness_value": np.nan})
t3d = pd.DataFrame(srows)
print("\nSensitivity:\n" + t3d.to_string(index=False))
save_table(t3d, "T3d_sensitivity")

# ---- 5. DoWhy refuters on a simple backdoor estimate (click outcome) ----
try:
    from dowhy import CausalModel
    cm = CausalModel(data=s[X + [S.TREATMENT, "click4"]].astype(float), treatment=S.TREATMENT, outcome="click4",
                     common_causes=X)
    estimand = cm.identify_effect(proceed_when_unidentifiable=True)
    est = cm.estimate_effect(estimand, method_name="backdoor.linear_regression")
    ref_rows = [{"refuter": "estimate (linear backdoor)", "value": round(est.value, 4), "new_effect": np.nan, "p": np.nan}]
    for meth in ["placebo_treatment_refuter", "random_common_cause"]:
        r = cm.refute_estimate(estimand, est, method_name=meth,
                               num_simulations=20 if DEMO else 100, random_seed=S.SEED)
        pv = (r.refutation_result or {}).get("p_value", np.nan) if isinstance(r.refutation_result, dict) else np.nan
        ref_rows.append({"refuter": meth, "value": round(est.value, 4), "new_effect": round(float(r.new_effect), 4),
                         "p": round(float(pv), 4) if pv == pv else np.nan})
    t3e = pd.DataFrame(ref_rows)
except Exception as e:                           # noqa: BLE001 - keep the run going, say why
    t3e = pd.DataFrame([{"refuter": f"DoWhy not run: {type(e).__name__}: {e}", "value": np.nan, "new_effect": np.nan, "p": np.nan}])
print("\nDoWhy refuters (placebo new_effect should be near 0, with a small p; random common cause should leave\n"
      "the estimate almost unchanged, with a large p):\n"
      + t3e.to_string(index=False))
save_table(t3e, "T3e_refuters")

# ---- 6. exploratory causal forest (EconML): distribution of individual estimates ----
try:
    from econml.dml import CausalForestDML
    cf = CausalForestDML(model_y=LEARNERS["boosting"](), model_t=LEARNERS["boosting"](), discrete_treatment=True,
                         discrete_outcome=True, n_estimators=200 if DEMO else 2000, min_samples_leaf=20,
                         honest=True, random_state=S.SEED, cv=S.N_FOLDS)
    cf.fit(s["click4"].to_numpy(), s[S.TREATMENT].to_numpy(), X=s[X].to_numpy(float))
    tau = np.ravel(cf.effect(s[X].to_numpy(float)))
    q = np.percentile(tau, [5, 25, 50, 75, 95])
    t3f = pd.DataFrame([{"statistic": n, "value": round(v, 4)} for n, v in
                        zip(["p5", "p25", "median", "p75", "p95"], q)] +
                       [{"statistic": "mean (compare with ATE)", "value": round(tau.mean(), 4)}])
    pd.DataFrame({"tau": tau}).to_csv(S.TABLES / ("REHEARSAL_forest_tau.csv" if DEMO else "forest_tau.csv"), index=False)
except Exception as e:                           # noqa: BLE001
    t3f = pd.DataFrame([{"statistic": f"EconML not run: {type(e).__name__}: {e}", "value": np.nan}])
print("\nCausal forest, exploratory only (never use single-person estimates to target anyone):\n" + t3f.to_string(index=False))
save_table(t3f, "T3f_forest_cate")
done("python 04_analysis/08_figures.py   (then Rscript 04_analysis/07_invariance.R)")

"""Step 04: rebuild the baseline's H2 path model (and the H1 total effect) and compare with the paper.
Run:     python 04_analysis/04_benchmark_fiml.py
Output:  05_results/tables/T1_replication.csv
Method:  joint FIML with random covariates, robust sandwich SEs (fiml.py). In the demo the
         "baseline" column is still the paper's value, so most rows will say NOT REPLICATED.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import settings as S
from common import banner, done, load_staff, save_table
from fiml import PathModel

banner("Step 04", "baseline rebuild (benchmark)", ["T1_replication"])
d = load_staff()
s = d[d[f"rule_{S.SAMPLE_RULE_BENCHMARK}"] == 1]
cols = ["lvl3"] + S.CONTROLS_BENCHMARK + S.MEDIATORS + ["training_intention"]
Z = s[cols].to_numpy(float)
print(f"Fitting the H2 model on N = {len(s)} (rows with missing covariates kept, as in the paper) ...")
m1 = PathModel(Z, free_med_cov=False).fit()
m2 = PathModel(Z, free_med_cov=True).fit()
chi2 = 2 * (m1.fun - m2.fun)
V = m1.robust_cov()
th = m1.theta
a_std, b_std, c_std = m1.std_paths()
B = S.BASELINE_H2
rows = [{"quantity": "H2 chi-square (df 15)", "rebuilt": round(chi2, 2), "baseline": B["chi2"]}]
g_tot = np.zeros(m1.n_par)
for j, med in enumerate(S.MEDIATORS):
    ia, ib = m1.index_a(j), m1.index_b(j)
    a, b = th[ia], th[ib]
    se = np.sqrt(b * b * V[ia, ia] + a * a * V[ib, ib] + 2 * a * b * V[ia, ib])
    g_tot[ia] += b; g_tot[ib] += a
    rows += [{"quantity": f"a path (std): {med}", "rebuilt": round(a_std[j], 3), "baseline": B["a_std"][med]},
             {"quantity": f"b path (std): {med}", "rebuilt": round(b_std[j], 3), "baseline": B["b_std"][med]},
             {"quantity": f"indirect B: {med}", "rebuilt": round(a * b, 3), "baseline": B["ind_B"][med]},
             {"quantity": f"indirect SE: {med}", "rebuilt": round(se, 3), "baseline": B["ind_SE"][med]}]
tot_ind = sum(th[m1.index_a(j)] * th[m1.index_b(j)] for j in range(6))
rows += [{"quantity": "total indirect B", "rebuilt": round(tot_ind, 3), "baseline": B["total_indirect"][0]},
         {"quantity": "total indirect SE", "rebuilt": round(np.sqrt(g_tot @ V @ g_tot), 3), "baseline": B["total_indirect"][1]}]
g_all = g_tot.copy(); g_all[m1.index_c()] = 1
rows += [{"quantity": "total effect B", "rebuilt": round(th[m1.index_c()] + tot_ind, 3), "baseline": B["total"][0]},
         {"quantity": "total effect SE", "rebuilt": round(np.sqrt(g_all @ V @ g_all), 3), "baseline": B["total"][1]}]
# CR2: residual covariances freed (rigour fix), indirect effects only
for j, med in enumerate(S.MEDIATORS):
    rows.append({"quantity": f"CR2 indirect B (residual covariances freed): {med}",
                 "rebuilt": round(m2.theta[m2.index_a(j)] * m2.theta[m2.index_b(j)], 3), "baseline": np.nan})
# H1 total effect: rows with complete covariates, as lavaan's fixed.x default did (n = 934 in the paper)
h1 = smf.ols("lvl4 ~ lvl3 + age_num + gender_bin + jobrole_bin01 + years_org_num", s).fit(cov_type="HC1")
rows += [{"quantity": f"H1 total effect B (n = {int(h1.nobs)})", "rebuilt": round(h1.params["lvl3"], 3), "baseline": S.BASELINE_H1_TOTAL[0]},
         {"quantity": "H1 total effect SE", "rebuilt": round(h1.bse["lvl3"], 3), "baseline": S.BASELINE_H1_TOTAL[1]}]
t1 = pd.DataFrame(rows)
tol = np.where(t1["quantity"].str.startswith("H2 chi"), 1.0, S.CLOSE_ENOUGH)
t1["status"] = np.where(t1["baseline"].isna(), "new (no baseline value)",
                        np.where((t1["rebuilt"] - t1["baseline"]).abs() <= tol, "REPLICATED", "NOT REPLICATED"))
print(t1.to_string(index=False))
print(f"\nReplicated: {(t1.status == 'REPLICATED').sum()} of {(t1.status != 'new (no baseline value)').sum()} comparable rows")
save_table(t1, "T1_replication")
done("python 04_analysis/05_measurement.py")

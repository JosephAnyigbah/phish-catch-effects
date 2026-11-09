"""Step 03: screening and descriptives for both sample rules.
Run:     python 04_analysis/03_screening.py
Output:  05_results/tables/T0_sample_and_rates.csv, T0b_scale_missingness.csv, T5a_correlations.csv
"""
import numpy as np
import pandas as pd

import settings as S
from common import banner, done, load_staff, save_table

banner("Step 03", "screening and descriptives", ["T0_sample_and_rates", "T0b_scale_missingness", "T5a_correlations"])
d = load_staff()
rows = []
for rule in ["published", "corrected"]:
    s = d[d[f"rule_{rule}"] == 1]
    row = {"rule": rule, "N": len(s)}
    for v in ["click3", "comp3", "report3", "click4", "comp4", "report4"]:
        row[f"{v}_%"] = round(100 * s[v].mean(), 1)
    row["history_known"] = int((s["history_missing"] == 0).sum())
    rows.append(row)
t0 = pd.DataFrame(rows)
print(t0.to_string(index=False))
save_table(t0, "T0_sample_and_rates")

s = d[d[f"rule_{S.SAMPLE_RULE_BENCHMARK}"] == 1]
miss = pd.DataFrame({"scale": S.MEDIATORS + ["training_intention"],
                     "n_complete": [int(s[c].notna().sum()) for c in S.MEDIATORS + ["training_intention"]]})
miss["n_missing_by_design"] = len(s) - miss["n_complete"]
print("\nScale completeness (planned missingness):\n" + miss.to_string(index=False))
save_table(miss, "T0b_scale_missingness")

cols = S.MEDIATORS + ["training_intention"]
desc = s[cols].agg(["mean", "std", "count"]).T.round(3)
corr = s[cols].corr().round(2)
out = pd.concat([desc, corr], axis=1).reset_index().rename(columns={"index": "scale"})
save_table(out, "T5a_correlations")

small = [g for g in ["role", "age3", "prior"] for k, n in s.groupby(g)["caught3"].sum().items() if n < S.MIN_CELL]
if small:
    print(f"\nNOTE: some treated cells are below {S.MIN_CELL}. Merge groups before publishing: {small}")
done("python 04_analysis/04_benchmark_fiml.py")

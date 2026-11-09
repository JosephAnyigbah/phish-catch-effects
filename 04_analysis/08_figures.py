"""Step 08: figures (PNG for drafts, PDF for the journal).
Run:     python 04_analysis/08_figures.py
Output:  Fig1_design, Fig2_group_effects, Fig3_replication, Fig4_rates, Fig5_forest_spread (if step 06 ran EconML)
Every demo figure carries the watermark "SYNTHETIC DEMO DATA - not a result".
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

import settings as S
from common import banner, done, tag

DEMO = S.DATA_MODE == "rehearsal"
C = S.PALETTE
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})


def finish(fig, name):
    if DEMO:
        fig.text(0.5, 0.5, S.WATERMARK, fontsize=20, color="grey", alpha=0.25, ha="center", va="center", rotation=25)
    S.FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(S.FIGURES / f"{tag(name)}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> 05_results/figures/{tag(name)}.png (+ .pdf)")


def table(name):
    p = S.TABLES / f"{tag(name)}.csv"
    if not p.exists():
        raise SystemExit(f"Missing {p.name}. Fix: run the step that writes it first.")
    return pd.read_csv(p)


banner("Step 08", "figures", ["Fig1 to Fig5 in 05_results/figures"])

# Fig 1: study design (a plain box diagram; no data)
fig, ax = plt.subplots(figsize=(9, 3.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
boxes = {"Before Sim 3\nage, gender, role, tenure\nSims 1-2 clicks": (0.2, 1.4),
         "Caught in Sim 3\n(clicked; page shown\nif credentials entered)": (3.6, 1.4),
         "Sim 4 behaviour\nclick, compromise,\nreport (logs)": (7.0, 1.4)}
for txt, (x, y) in boxes.items():
    ax.add_patch(FancyBboxPatch((x, y), 2.8, 1.3, boxstyle="round,pad=0.05", fc="white", ec=C[0], lw=1.6))
    ax.text(x + 1.4, y + 0.65, txt, ha="center", va="center")
for x0, x1 in [(3.0, 3.6), (6.4, 7.0)]:
    ax.annotate("", xy=(x1, 2.05), xytext=(x0, 2.05), arrowprops=dict(arrowstyle="->", color=C[0], lw=1.6))
ax.annotate("", xy=(7.1, 2.75), xytext=(1.6, 2.75), arrowprops=dict(arrowstyle="->", color=C[1], lw=1.2,
            connectionstyle="arc3,rad=-0.25"))
ax.text(4.9, 3.75, "adjusted for (double machine learning)", ha="center", color=C[1])
ax.text(5.0, 0.5, "Groups: role, age band, prior clicking.   Survey scales (after Sim 3) are never used as controls.",
        ha="center", fontsize=9)
ax.set_title("Figure 1. Study design")
finish(fig, "Fig1_design")

# Fig 2: group effects with 95% CIs
g = table("T3_gates")
fig, ax = plt.subplots(figsize=(7, 4.2))
ypos = range(len(g))[::-1]
colors = [C[["role", "age3", "prior"].index(k)] for k in g["grouping"]]
ax.errorbar(g["effect"], list(ypos), xerr=[g["effect"] - g["ci_low"], g["ci_high"] - g["effect"]],
            fmt="none", ecolor=colors, elinewidth=2)
ax.scatter(g["effect"], list(ypos), c=colors, zorder=3)
ax.axvline(0, color="grey", lw=1, ls="--")
ax.set_yticks(list(ypos)); ax.set_yticklabels([f"{a}: {b}" for a, b in zip(g["grouping"], g["group"])])
ax.set_xlabel("Effect of being caught on Sim 4 clicking (risk difference, 95% CI)")
ax.set_title("Figure 2. Group effects")
finish(fig, "Fig2_group_effects")

# Fig 3: replication of the baseline paths
t1 = table("T1_replication")
p = t1[t1["quantity"].str.contains(r"path \(std\)")]
fig, ax = plt.subplots(figsize=(4.8, 4.6))
ax.scatter(p["baseline"], p["rebuilt"], color=C[0])
lim = [min(p["baseline"].min(), p["rebuilt"].min()) - 0.05, max(p["baseline"].max(), p["rebuilt"].max()) + 0.05]
ax.plot(lim, lim, color="grey", ls="--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Baseline (paper, Fig 3)"); ax.set_ylabel("Rebuilt")
ax.set_title("Figure 3. Rebuilt against reported paths")
finish(fig, "Fig3_replication")

# Fig 4: click, compromise and report rates by sample rule
t0 = table("T0_sample_and_rates")
fig, ax = plt.subplots(figsize=(7, 3.6))
cols = ["click3_%", "comp3_%", "report3_%", "click4_%", "comp4_%", "report4_%"]
w = 0.38
for i, (_, r) in enumerate(t0.iterrows()):
    ax.bar([k + i * w for k in range(len(cols))], r[cols], width=w, color=C[i], label=f"{r['rule']} rule (N = {r['N']})")
ax.set_xticks([k + w / 2 for k in range(len(cols))]); ax.set_xticklabels([c.replace("_%", "") for c in cols])
ax.set_ylabel("Per cent of staff"); ax.legend(frameon=False); ax.set_title("Figure 4. Behaviour in Simulations 3 and 4")
finish(fig, "Fig4_rates")

# Fig 5: spread of causal-forest estimates (exploratory)
tp = S.TABLES / ("REHEARSAL_forest_tau.csv" if DEMO else "forest_tau.csv")
if tp.exists():
    tau = pd.read_csv(tp)["tau"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(tau, bins=40, color=C[2]); ax.axvline(tau.mean(), color=C[0], lw=2, label="mean")
    ax.set_xlabel("Estimated effect for each staff member (exploratory only)"); ax.set_ylabel("Staff")
    ax.legend(frameon=False); ax.set_title("Figure 5. Spread of causal-forest estimates")
    finish(fig, "Fig5_forest_spread")
else:
    print("  Fig5 skipped: step 06 did not write forest estimates (EconML missing?)")
done("Rscript 04_analysis/07_invariance.R   (if not run yet)")

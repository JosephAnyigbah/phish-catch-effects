"""Run the whole analysis on REHEARSAL data (no real respondents). This is your first milestone.
Run:     python run_demo.py
Takes:   about 1 to 2 minutes on a laptop CPU.
Output:  05_results/tables/REHEARSAL_*.csv and 05_results/figures/REHEARSAL_*.png/.pdf
Delete the REHEARSAL_ outputs before your real run (python run_demo.py --clean).
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
A = ROOT / "04_analysis"
STEPS = ["01_rehearsal_data.py", "03_screening.py", "04_benchmark_fiml.py", "05_measurement.py",
         "06_causal_dml.py", "08_figures.py"]


def clean():
    n = 0
    for folder in [ROOT / "05_results" / "tables", ROOT / "05_results" / "figures"]:
        for f in folder.glob("REHEARSAL_*"):
            f.unlink(); n += 1
    print(f"Removed {n} rehearsal output file(s).")


if "--clean" in sys.argv:
    clean(); sys.exit(0)

text = (A / "settings.py").read_text()
if 'DATA_MODE = "rehearsal"' not in text:
    sys.exit('The demo needs DATA_MODE = "rehearsal" in 04_analysis/settings.py. Fix that line and run again.')
clean()
t0 = time.time()
for step in STEPS:
    print(f"\n>>> python 04_analysis/{step}")
    r = subprocess.run([sys.executable, str(A / step)], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"\nThe demo stopped at {step}. Read the message above, fix it, then run python run_demo.py again.")
rscript = shutil.which("Rscript")
if rscript:
    print("\n>>> Rscript 04_analysis/07_invariance.R")
    r = subprocess.run([rscript, str(A / "07_invariance.R")], cwd=ROOT)
    if r.returncode != 0:
        print("The R step failed. The Python steps are fine. See Appendix A, A16, for R fixes.")
else:
    print("\nRscript not found, so step 07 (invariance) was skipped. Install R (Appendix A, A4) or use Colab Part C.")
mins, secs = divmod(time.time() - t0, 60)
print(f"\nDEMO COMPLETE in {int(mins)} min {secs:.0f} s. Open 05_results/figures and 05_results/tables.")
print("Every number above is SYNTHETIC DEMO DATA - not a result.")

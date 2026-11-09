"""Step 00: check Python, packages, memory and disk.
Run:     python 04_analysis/00_check_env.py
Output:  printed report only. Fix anything marked MISSING before step 01.
"""
import importlib
import platform
import shutil
import sys
import warnings

warnings.filterwarnings("ignore")      # some packages print harmless warnings the first time they load

NEEDED = ["numpy", "pandas", "scipy", "statsmodels", "sklearn", "doubleml", "econml",
          "dowhy", "semopy", "matplotlib", "openpyxl"]

print("Python", platform.python_version(), "on", platform.system(), platform.machine())
if sys.version_info < (3, 11):
    print("PROBLEM: Python 3.11 or later is needed. Install a newer Python (Appendix A, A4).")
bad = 0
for name in NEEDED:
    try:
        mod = importlib.import_module(name)
        print(f"  ok       {name:12s} {getattr(mod, '__version__', '?')}")
    except Exception as e:                       # noqa: BLE001 - show any import failure plainly
        bad += 1
        print(f"  MISSING  {name:12s} -> run: pip install -r requirements.txt   ({type(e).__name__})")
try:
    import os
    mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    print(f"RAM: {mem_gb:.1f} GB (4 GB is enough)")
except (ValueError, AttributeError, OSError):
    print("RAM: could not read on this system (4 GB is enough)")
free = shutil.disk_usage(".").free / 1e9
print(f"Free disk: {free:.1f} GB (1 GB is enough)")
print("\nAll packages found. Next: python run_demo.py" if bad == 0 else f"\n{bad} package(s) missing. Fix them first.")

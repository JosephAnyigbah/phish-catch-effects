"""Small helpers shared by the numbered scripts."""
import re
import sys
import numpy as np
import pandas as pd
import settings as S


def banner(step, what, outputs):
    print("=" * 72)
    print(f"{step}: {what}")
    print(f"Mode: {S.DATA_MODE.upper()}" + ("   (" + S.WATERMARK + ")" if S.DATA_MODE == "rehearsal" else ""))
    print("Writes: " + ", ".join(outputs))
    print("=" * 72)


def done(next_step):
    print(f"\nDone. Next: {next_step}\n")


def tag(name):
    """File names carry REHEARSAL in demo mode so demo output is never mistaken for a result."""
    return f"REHEARSAL_{name}" if S.DATA_MODE == "rehearsal" else name


def load_staff():
    path = (S.REHEARSAL_DIR / "REHEARSAL_staff.csv") if S.DATA_MODE == "rehearsal" else (S.CLEAN_DIR / "staff_clean.csv")
    if not path.exists():
        sys.exit(f"Missing {path}.\nFix: run 01_rehearsal_data.py (demo) or 02_fetch_import.py (real data) first.")
    return pd.read_csv(path)


def save_table(df, name):
    S.TABLES.mkdir(parents=True, exist_ok=True)
    out = S.TABLES / f"{tag(name)}.csv"
    df.to_csv(out, index=False)
    print(f"  table -> {out.relative_to(S.ROOT)}")
    return out


def leading_number(x):
    """'7. Strongly agree' -> 7.0, '1 (very low)' -> 1.0, 4.0 -> 4.0, '' or NaN -> NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)", str(x))
    return float(m.group(1)) if m else np.nan


def event(series):
    """Log cell -> 1 if an event happened, 0 if '--', NaN if the column is empty for that person."""
    miss = series.isna()
    t = series.astype(str).str.strip()
    out = pd.Series(np.where(t == "--", 0.0, 1.0), index=series.index)
    out[miss | (t == "")] = np.nan
    return out


def parse_timestamp(x):
    """Handles both formats found in the Sim 4 columns: Excel serial numbers and dd/mm/yyyy text.

    The serial numbers were created when a spreadsheet read 12/05/2025 as 5 December.
    So a serial date whose day is 12 or less has its day and month swapped back.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.NaT
    s = str(x).strip()
    if s in ("", "--"):
        return pd.NaT
    if re.fullmatch(r"\d{5}(\.\d+)?", s):                       # Excel serial day number
        t = pd.Timestamp("1899-12-30") + pd.to_timedelta(float(s), unit="D")
        if t.day <= 12:
            t = t.replace(month=t.day, day=t.month)
        return t
    if re.match(r"\d{2}/\d{2}/\d{4}", s):                       # day first, as in the export
        return pd.to_datetime(s, dayfirst=True, errors="coerce")
    return pd.to_datetime(s, errors="coerce")

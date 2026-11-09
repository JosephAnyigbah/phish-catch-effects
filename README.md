# phish-catch-effects

Who is helped or harmed by being caught in a phishing simulation? This repository rebuilds the
open data of George et al. (2027, IEEE S&P, data: https://doi.org/10.5281/zenodo.18503743),
estimates the effect of being caught on later clicking, compromise and reporting with double
machine learning, and tests five staff security scales between that university and KNUST.

Group: Nahimatu Ayippey (causal arm), Joseph Edem Anyigbah (benchmark model), Grace Badu Gyimah
(KNUST survey and invariance). Supervisor: Dr Eric Opoku Osei.

## Quick start (demo on rehearsal data, about 1 to 2 minutes)

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS or Linux
pip install -r requirements.txt
python 04_analysis/00_check_env.py
python -m pytest -q
python run_demo.py
```

Every demo number is **SYNTHETIC DEMO DATA - not a result**. Remove demo outputs with
`python run_demo.py --clean` before the real run. Full instructions: Appendix A (Testbed Guide).

## Step table

| Step | Script | What it does | Writes | Answers |
|---|---|---|---|---|
| 00 | `04_analysis/00_check_env.py` | Checks Python, packages, memory and disk | screen only | setup |
| 01 | `04_analysis/01_rehearsal_data.py` | Builds rehearsal data from the paper's numbers | `03_data/rehearsal/` | demo |
| 02 | `04_analysis/02_fetch_import.py` | Fetches the Zenodo zip, checks its md5, builds the clean table and both sample rules | `03_data/clean/staff_clean.csv` | all |
| 02b | `04_analysis/02b_import_knust.py` | Imports and screens the KNUST survey export | `03_data/clean/knust_clean.csv` | RQ2 |
| 03 | `04_analysis/03_screening.py` | Sample sizes, behaviour rates, planned missingness, correlations | T0, T0b, T5a | all |
| 04 | `04_analysis/04_benchmark_fiml.py` | Rebuilds the baseline path model (FIML, robust SEs) and marks each path replicated or not | T1, Fig 3 | RQ1 benchmark |
| 05 | `04_analysis/05_measurement.py` | CFA per scale, reliability, AVE, HTMT with bootstrap CIs | T6, T6b | RQ2 |
| 06 | `04_analysis/06_causal_dml.py` | Average effects, group effects, heterogeneity test, page contrast, sensitivity, refuters, causal forest | T2, T3 to T3f, Fig 2, Fig 5 | RQ1, H1 to H4 |
| 07 | `04_analysis/07_invariance.R` | Configural, metric and scalar invariance (lavaan, semTools) | T4 | RQ2, H5 |
| 08 | `04_analysis/08_figures.py` | Figures 1 to 5 (PNG and PDF) | `05_results/figures/` | all |

All settings live in `04_analysis/settings.py`. No script hard-codes a setting.

The respondent-ready KNUST questionnaire, with the participant introduction and five consent
boxes, is `02_instrument/KNUST_Staff_Questionnaire.docx`. A form-builder copy is
`02_instrument/knust_survey_items.csv` (question names match `02_instrument/knust_codebook.csv`).
Response formats: 7-point Likert-type scales for sections B, C, D and F, a 7-point bipolar scale for
section E, single choice for section A, and yes/no only for the skip question. Step 02b drops any row
without all five consent boxes ticked. The rules for asking and collecting it are in the Survey Kit.

## Rules

- Never commit raw data, KNUST responses, identifiers, letters or consent forms (`.gitignore` blocks them).
- The open-data file is fetched from Zenodo by script. It is never copied into this repository.
- Never paste survey responses, participant data or keys into any AI service. Log every AI use in `06_docs/AI_USE_LOG.md`.
- Adjust only for variables measured before Simulation 3. The survey scales are never controls.
- Post the preregistration (`06_docs/PREREGISTRATION.md`) on OSF before setting `DATA_MODE = "real"`.
- Publish no table cell with fewer than 10 staff.

## Licences

Code: MIT (see LICENSE). Open data: CC BY 4.0, George et al. Scale items: see
`02_instrument/construct_item_map.md` for each source and its permission status.

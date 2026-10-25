# ImariShield: explainable financial resilience for Rwanda

**NISR 2026 Big Data Hackathon · Track 2: Financial Inclusion & Poverty Reduction · Team G29 (African Leadership University)**

🔗 **Live app:** `<paste Streamlit link here>`  ·  🎬 **Demo video:** `demo/imarishield_demo.webm`

## The problem
Access to financial services is not the same as resilience. In FinScope Rwanda 2024, about **76%** of adults could not cover four weeks of essentials from savings and about **82%** would find raising RWF 56,000 within a month very difficult. Programmes need to know *where* these adults are, *who* they are, and *what is associated* with better outcomes.

## What the app shows
| Page | What you get |
|---|---|
| Overview | Headline numbers and district comparison |
| District map | Choropleth of the 30 districts, priority ranking with intervals and adults affected, district profile |
| Who is most exposed | Gaps by sex, urban/rural, education, income source, work, Ubudehe, marital status, age |
| Risk calculator | Pick a profile → two stress-test probabilities, percentile, the answers that drive the estimate, "what if" associations |
| Model, fairness & limits | Test metrics, drivers, calibration, fairness audit, honest limitations |

## Data
FinScope Rwanda 2024 microdata (13,994 adults, survey weights `pop_wt`) from the NISR portal: https://microdata.statistics.gov.rw . District boundaries from geoBoundaries (CC BY 4.0). Details in [`data/README_DATA.md`](data/README_DATA.md). **Raw microdata is not redistributed**; this repo contains only aggregates and trained models.

## Method
- **Outcomes (from FinScope's own stress questions):** *E7d* could pay essentials for 4 weeks from savings; *E9c* could find RWF 56,000 within a month. "Don't know" answers are excluded (modelling sample 12,863).
- **Inputs:** 169 pre-shock characteristics. Every question that defines or follows the outcomes (E-block, C5 items) is excluded, enforced by an assertion in `train.py`.
- **Models:** XGBoost with survey weights and native categorical features (deployed); multi-task PyTorch network with embeddings as comparison (TensorBoard logs in `runs/`). Held-out test AUC: XGBoost 0.82 / 0.86, neural network 0.79 / 0.83.
- **Explainability:** SHAP-style contributions from the booster; association-based "levers".
- **Fairness:** performance by sex, area, education, Ubudehe and district.
- **Estimates:** weighted shares with bootstrap 95% intervals (these ignore survey clustering, so they are optimistic).

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```
Retrain (needs the microdata `.dta` from NISR):
```bash
pip install -r requirements-train.txt
python train.py --data path/to/FinScope_2024_Rwanda.dta      # add --skip-nn to skip the neural network
tensorboard --logdir runs
```

## Repository layout
```
app.py                 Streamlit dashboard
train.py               full pipeline: targets, features, leakage guard, models, estimates, fairness
artifacts/             trained XGBoost models, model spec, metrics
data/                  aggregate results, boundaries, data notes
demo/                  screen-recorded demo (≤3 min)
requirements*.txt
```

## Limits
Self-reported hypothetical outcomes; associations are not causal effects; intervals are optimistic; ~300–470 respondents per district; the model is less accurate for rural adults and in some districts. Use for **area-level planning, never for individual eligibility decisions**.

## Declarations
Intellectual property is assigned to NISR under the hackathon rules. AI assistance (Claude) was used for coding and drafting; the team is responsible for all analysis and claims. Licences: code MIT; boundaries CC BY 4.0.

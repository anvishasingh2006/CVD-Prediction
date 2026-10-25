# Explainable ML cardiovascular risk stratification in type 2 diabetes (NHANES)

Predicts **10-year cardiovascular death** in US adults with type 2 diabetes from structured clinical data, using only official CDC/NCHS data:
NHANES 1999-2008 examination, laboratory and questionnaire files, linked to the NCHS 2019 Public-Use Linked Mortality Files (follow-up to 31 Dec 2019).
Models: Logistic Regression, Random Forest, XGBoost (plus an age-only baseline), SHAP explanations, 3-tier risk stratification.

## Files (flat project)
| File | Purpose |
|---|---|
| `config.py` | All settings and the feature list |
| `utils.py` | Pipeline builder, SHAP helper, risk-tier helper |
| `build_cohort.py` | Downloads NHANES + mortality files, builds cohort, outcome, features, fixed train/test split |
| `eda.py` | Table 1 and EDA figures (training rows only) |
| `train_models.py` | Tuning, evaluation, calibration, risk tiers, saves models |
| `explain_shap.py` | Global and individual SHAP outputs |
| `predict_risk.py` | Scores new patients (CSV or built-in examples) with risk, tier and top drivers |
| `run_all.py` | Runs the five scripts in order |

Every script also works cell by cell in VS Code (`# %%` markers).

## Run
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py
```
The first run downloads about 60 small files from cdc.gov into `data/raw/` (needs internet) and caches them. Model tuning takes roughly 5-20 minutes depending on CPU cores.
Scripts can also be run one by one in the order above. If a download is blocked, save the file manually at the path shown in the error message and re-run.

Score your own patients: `python predict_risk.py patients.csv` where the CSV has columns from `FEATURES` in `config.py` (clinical units, `log_uacr` = natural log of UACR in mg/g; missing values are imputed with training medians).

## Outputs (created by the scripts)
- `data/processed/cohort.csv`: one row per participant, features, outcome, `split`
- `models/`: `logreg.joblib`, `rf.joblib`, `xgb.joblib`, `age_only.joblib`, `recalibrator.joblib`, `model_info.json`
- `outputs/tables/`: `cohort_flow.csv`, `data_sources.csv` (URL and SHA-256 of every raw file), `table1.csv`, `cv_summary.csv`, `test_metrics.csv`, `risk_tiers_test.csv`, `risk_tiers_train_oof.csv`, `test_predictions.csv`, `shap_global_importance.csv`, `shap_by_risk_tier.csv`, `shap_individual_examples.csv`, `patient_risk_scores.csv`
- `outputs/figures/`: `eda_*.png` (3), `roc_pr_curves.png`, `calibration_and_risk_tiers.png`, `shap_beeswarm.png`, `shap_bar.png`, `shap_dependence.png`, `shap_by_risk_tier.png`, `shap_waterfall_*.png` (4)

## Dataset URLs
- NHANES files: `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/<cycle start year>/DataFiles/<FILE>.xpt`
- Linked mortality: `https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_<y0>_<y1>_MORT_2019_PUBLIC.dat`
- Documentation: https://wwwn.cdc.gov/nchs/nhanes/ and https://www.cdc.gov/nchs/linked-data/

File names by cycle (names changed between 1999-2004 and 2005 onward):

| Component | 1999-2000 | 2001-2002 | 2003-2004 | 2005-2006 | 2007-2008 |
|---|---|---|---|---|---|
| DEMO | `DEMO` | `DEMO_B` | `DEMO_C` | `DEMO_D` | `DEMO_E` |
| BMX | `BMX` | `BMX_B` | `BMX_C` | `BMX_D` | `BMX_E` |
| BPX | `BPX` | `BPX_B` | `BPX_C` | `BPX_D` | `BPX_E` |
| SMQ | `SMQ` | `SMQ_B` | `SMQ_C` | `SMQ_D` | `SMQ_E` |
| DIQ | `DIQ` | `DIQ_B` | `DIQ_C` | `DIQ_D` | `DIQ_E` |
| BPQ | `BPQ` | `BPQ_B` | `BPQ_C` | `BPQ_D` | `BPQ_E` |
| MCQ | `MCQ` | `MCQ_B` | `MCQ_C` | `MCQ_D` | `MCQ_E` |
| GHB | `LAB10` | `L10_B` | `L10_C` | `GHB_D` | `GHB_E` |
| TCHOL | `LAB13` | `L13_B` | `L13_C` | `TCHOL_D` | `TCHOL_E` |
| HDL | `LAB13` | `L13_B` | `L13_C` | `HDL_D` | `HDL_E` |
| BIOPRO | `LAB18` | `L40_B` | `L40_C` | `BIOPRO_D` | `BIOPRO_E` |
| ALBCR | `LAB16` | `L16_B` | `L16_C` | `ALB_CR_D` | `ALB_CR_E` |

## NHANES variable mapping
| Feature | NHANES source |
|---|---|
| `age` (capped at 80), `female` | `RIDAGEYR`, `RIAGENDR` (2 = female) |
| `bmi`, `waist_cm` | `BMXBMI`, `BMXWAIST` |
| `sbp`, `dbp`, `pulse_pressure` | mean of `BPXSY2-4` / `BPXDI2-4` (reading 1 if none); `BPXDI = 0` treated as missing |
| `total_chol` | `LBXTC` (mg/dL) |
| `hdl_chol` | `LBDHDD`, `LBDHDL` or `LBXHDD` (name differs by cycle, mg/dL) |
| `non_hdl_chol`, `tc_hdl_ratio` | derived from the two above |
| `hba1c` | `LBXGH` (%) |
| `dm_duration_years`, `dm_diagnosed` | `RIDAGEYR` minus `DID040`; `DIQ010 == 1` |
| `insulin_use`, `oral_dm_med` | `DIQ050 == 1`, `DIQ070 == 1` |
| `htn_diagnosed`, `htn_med` | `BPQ020`, `BPQ050A` |
| `chol_med` | `BPQ100D` (not asked in 1999-2000, left missing for that cycle) |
| `smoker_current`, `smoker_former` | `SMQ020` with `SMQ040` (1-2 = current, 3 = former) |
| `egfr` | CKD-EPI 2021 (race-free) from serum creatinine `LBXSCR` / `LBDSCR` |
| `log_uacr`, `albuminuria` | `URXUMA / URXUCR * 100` (mg/g); albuminuria = UACR >= 30 |
| Exclusion: prior CVD | `MCQ160B-F == 1` (CHF, CHD, angina, MI, stroke) |
| Outcome | `MORTSTAT == 1`, `UCOD_LEADING` in (001 heart disease, 005 cerebrovascular), `PERMTH_EXM <= 120` months |

## Cohort and outcome
Adults >= 30 y, non-pregnant, MEC-examined, with type 2 diabetes (physician diagnosis or HbA1c >= 6.5%; probable type 1 removed: diagnosed < 30 y and on insulin), eligible for mortality linkage, no self-reported prior CVD.
Label = cardiovascular death within 10 years. Non-CV deaths inside the window are labelled 0 (cause-specific outcome). Survivors with < 10 years of follow-up are excluded, so every row has a fully observed 10-year outcome. This is why cycles stop at 2007-2008: later cycles have too little follow-up for a 10-year label (`build_cohort.py` stops with an error if a cycle cannot support the horizon).

## Leakage controls
- The train/test split (80/20, stratified, seed 42) is fixed in `build_cohort.py` before any modelling; EDA uses training rows only
- Imputation and scaling are inside each `Pipeline` and are fit on training folds only; tuning uses 5-fold CV on the training set
- Primary model = best training CV AUC (the test set never chooses anything)
- Decision thresholds, recalibration and risk-tier cut-points come from out-of-fold training predictions
- No mortality-file column is a feature (asserted), train/test participants are disjoint (asserted), and a shuffled-label model must score about 0.5 AUC (asserted)

## About the 75% target
`train_models.py` prints whether the primary model reaches `TARGET_AUC = 0.75` on the held-out test set, with a bootstrap CI and the cross-validated AUC. The target is a check, not something the code optimises. Nothing is tuned on the test set, and the numbers are whatever the real data give. With a few dozen test events the CI is wide (about +/- 0.05), so also look at `cv_auc`. If the result is below target, legitimate levers are: more events (more cycles with a shorter `HORIZON_YEARS`, or `MIN_AGE = 20`), or adding variables available at baseline. Do not include prior-CVD participants as a shortcut to a higher AUC unless you also change the question to "risk in all T2D patients".

## Validation status
The five scripts were run end to end on synthetic files that mimic the NHANES file/variable names and the mortality fixed-width layout, with each of the three models forced as primary (checks code paths, SHAP additivity, saved-model reload). That does not test performance on real data, and no numbers in this folder come from NHANES.

## Limitations
Few hundred CV deaths at most; training ignores NHANES survey weights and clustering (not a nationally representative estimate); serum creatinine and urine albumin assays changed across cycles and are not bridged (eGFR/UACR are best treated as exploratory); linked mortality captures fatal CVD only; SHAP shows model behaviour, not causal effects; research use only.

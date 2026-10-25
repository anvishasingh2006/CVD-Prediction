import json
import sys

import joblib
import numpy as np
import pandas as pd

from config import *
from utils import assign_tier, explain, logit

# %% usage: python predict_risk.py [patients.csv]
# CSV columns = FEATURES in config.py (clinical units). Missing columns/values are imputed with training medians.
EXAMPLE = pd.DataFrame([
    {'age': 45, 'female': 1, 'bmi': 29, 'waist_cm': 98, 'sbp': 122, 'dbp': 78, 'total_chol': 185, 'hdl_chol': 52,
     'hba1c': 6.8, 'dm_duration_years': 3, 'dm_diagnosed': 1, 'insulin_use': 0, 'oral_dm_med': 1, 'htn_diagnosed': 0,
     'htn_med': 0, 'chol_med': 0, 'smoker_current': 0, 'smoker_former': 0, 'egfr': 98, 'log_uacr': np.log(8)},
    {'age': 63, 'female': 0, 'bmi': 32, 'waist_cm': 109, 'sbp': 142, 'dbp': 82, 'total_chol': 205, 'hdl_chol': 40,
     'hba1c': 8.1, 'dm_duration_years': 11, 'dm_diagnosed': 1, 'insulin_use': 0, 'oral_dm_med': 1, 'htn_diagnosed': 1,
     'htn_med': 1, 'chol_med': 1, 'smoker_current': 1, 'smoker_former': 0, 'egfr': 72, 'log_uacr': np.log(45)},
    {'age': 76, 'female': 0, 'bmi': 28, 'waist_cm': 106, 'sbp': 158, 'dbp': 74, 'total_chol': 190, 'hdl_chol': 36,
     'hba1c': 8.9, 'dm_duration_years': 22, 'dm_diagnosed': 1, 'insulin_use': 1, 'oral_dm_med': 0, 'htn_diagnosed': 1,
     'htn_med': 1, 'chol_med': 1, 'smoker_current': 0, 'smoker_former': 1, 'egfr': 48, 'log_uacr': np.log(320)},
])


def prepare(raw):
    X = raw.reindex(columns=FEATURES).astype(float)
    X['pulse_pressure'] = X.pulse_pressure.fillna(X.sbp - X.dbp)
    X['non_hdl_chol'] = X.non_hdl_chol.fillna(X.total_chol - X.hdl_chol)
    X['tc_hdl_ratio'] = X.tc_hdl_ratio.fillna(X.total_chol / X.hdl_chol)
    X['albuminuria'] = X.albuminuria.fillna((np.exp(X.log_uacr) >= 30).astype(float).where(X.log_uacr.notna()))
    return X


def score(raw):
    info = json.load(open(MODEL_DIR / 'model_info.json'))
    pipe = joblib.load(MODEL_DIR / f"{info['primary_model']}.joblib")
    recal = joblib.load(MODEL_DIR / 'recalibrator.joblib')
    X = prepare(raw)
    p = pipe.predict_proba(X)[:, 1]
    risk = recal.predict_proba(logit(p).reshape(-1, 1))[:, 1]
    out = pd.DataFrame({'risk_10y_cv_death': risk,
                        'tier': assign_tier(risk, info['tier_cutpoints'], info['tier_labels'])}, index=raw.index)
    return out, X, pipe


if __name__ == '__main__':
    raw = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else EXAMPLE
    out, X, pipe = score(raw)
    background = pd.read_csv(PROC_DIR / 'cohort.csv').query("split == 'train'")[FEATURES]
    sv, unit = explain(pipe, X, background)
    drivers = []
    for i in range(len(X)):
        top = np.argsort(-np.abs(sv.values[i]))[:3]
        drivers.append('; '.join(f"{FEATURES[j]}={sv.data[i, j]:.1f} ({'+' if sv.values[i, j] > 0 else '-'})" for j in top))
    out['top_drivers'] = drivers
    out.to_csv(TAB_DIR / 'patient_risk_scores.csv', index=False)
    pd.set_option('display.width', 200, 'display.max_colwidth', 120)
    print(out.assign(risk_10y_cv_death=(out.risk_10y_cv_death * 100).round(1).astype(str) + '%').to_string())
    print(f'\n(+) raises / (-) lowers predicted risk; SHAP units: {unit}. Saved {TAB_DIR / "patient_risk_scores.csv"}')

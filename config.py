from pathlib import Path

SEED = 42
CYCLES = ['1999-2000', '2001-2002', '2003-2004', '2005-2006', '2007-2008']
HORIZON_YEARS = 10
MIN_AGE = 30
EXCLUDE_PREVALENT_CVD = True
TEST_SIZE = 0.20
CV_FOLDS = 5
N_ITER = {'logreg': 20, 'rf': 25, 'xgb': 50}
N_BOOT = 1000
TARGET_AUC = 0.75
TIER_QUANTILES = (0.50, 0.80)
TIER_LABELS = ['Low', 'Intermediate', 'High']
CV_DEATH_CODES = (1, 5)

ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
RAW_DIR = ROOT / 'data' / 'raw'
PROC_DIR = ROOT / 'data' / 'processed'
MODEL_DIR = ROOT / 'models'
FIG_DIR = ROOT / 'outputs' / 'figures'
TAB_DIR = ROOT / 'outputs' / 'tables'
for d in (RAW_DIR, PROC_DIR, MODEL_DIR, FIG_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)

FEATURES = ['age', 'female', 'bmi', 'waist_cm', 'sbp', 'dbp', 'pulse_pressure',
            'total_chol', 'hdl_chol', 'non_hdl_chol', 'tc_hdl_ratio',
            'hba1c', 'dm_duration_years', 'dm_diagnosed', 'insulin_use', 'oral_dm_med',
            'htn_diagnosed', 'htn_med', 'chol_med', 'smoker_current', 'smoker_former',
            'egfr', 'log_uacr', 'albuminuria']

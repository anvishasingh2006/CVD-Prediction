import subprocess
import sys

from config import ROOT

for script in ['build_cohort.py', 'eda.py', 'train_models.py', 'explain_shap.py', 'predict_risk.py']:
    print(f'\n===== {script} =====', flush=True)
    subprocess.run([sys.executable, '-u', script], cwd=ROOT, check=True)

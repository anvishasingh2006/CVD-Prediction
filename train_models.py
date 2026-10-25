import json
import time
import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_val_score

from config import *
from utils import assign_tier, logit, make_pipe

warnings.filterwarnings('ignore')
np.random.seed(SEED)

# %% data (split was fixed in build_cohort.py)
df = pd.read_csv(PROC_DIR / 'cohort.csv')
train, test = df[df.split == 'train'], df[df.split == 'test']
X_train, y_train = train[FEATURES], train.event
X_test, y_test = test[FEATURES], test.event
yt = y_test.values
print(f'train n={len(train)} events={y_train.sum()} | test n={len(test)} events={yt.sum()}')

# %% leakage checks
banned = ('mort', 'ucod', 'permth', 'death', 'event', 'status', 'eligstat', 'split')
assert not [f for f in FEATURES if any(b in f.lower() for b in banned)]
assert set(train.SEQN).isdisjoint(test.SEQN)

# %% models and tuning (imputer/scaler live inside the pipeline, so they are fit on training folds only)
cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
SPECS = {
    'logreg': (make_pipe(LogisticRegression(max_iter=5000, random_state=SEED), scale=True),
               {'clf__C': loguniform(1e-3, 1e2)}),
    'rf': (make_pipe(RandomForestClassifier(n_jobs=1, random_state=SEED)),
           {'clf__n_estimators': randint(200, 700), 'clf__max_depth': [3, 4, 5, 6, 8, None],
            'clf__min_samples_leaf': randint(5, 40), 'clf__max_features': ['sqrt', 0.3, 0.5]}),
    'xgb': (make_pipe(xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', tree_method='hist',
                                        n_jobs=1, random_state=SEED, verbosity=0)),
            {'clf__n_estimators': randint(100, 600), 'clf__learning_rate': loguniform(0.01, 0.2),
             'clf__max_depth': randint(2, 5), 'clf__min_child_weight': randint(1, 15),
             'clf__subsample': uniform(0.6, 0.4), 'clf__colsample_bytree': uniform(0.5, 0.5),
             'clf__reg_lambda': loguniform(1, 100), 'clf__reg_alpha': loguniform(1e-3, 10),
             'clf__gamma': uniform(0, 2)}),
}
NAMES = {'logreg': 'Logistic Regression', 'rf': 'Random Forest', 'xgb': 'XGBoost', 'age_only': 'Baseline: age only'}


def clean(d):
    return {k.replace('clf__', ''): (round(float(v), 4) if isinstance(v, (float, np.floating)) else
                                     int(v) if isinstance(v, (int, np.integer)) else v) for k, v in d.items()}


models, cv_rows = {}, []
for key, (pipe, space) in SPECS.items():
    t0 = time.time()
    s = RandomizedSearchCV(pipe, space, n_iter=N_ITER[key], scoring='roc_auc', cv=cv, random_state=SEED, n_jobs=-1)
    s.fit(X_train, y_train)
    models[key] = s.best_estimator_
    cv_rows.append({'model': key, 'cv_auc': s.best_score_, 'cv_auc_sd': s.cv_results_['std_test_score'][s.best_index_],
                    'best_params': json.dumps(clean(s.best_params_))})
    print(f'{NAMES[key]:<20} CV AUC {s.best_score_:.3f}  ({time.time() - t0:.0f}s)  {clean(s.best_params_)}')
cv_table = pd.DataFrame(cv_rows)
cv_table.to_csv(TAB_DIR / 'cv_summary.csv', index=False)
PRIMARY = cv_table.loc[cv_table.cv_auc.idxmax(), 'model']
print(f'\nPrimary model (best training CV AUC): {NAMES[PRIMARY]}')

models['age_only'] = make_pipe(LogisticRegression(max_iter=5000), scale=True).fit(X_train[['age']], y_train)
cols = {k: (['age'] if k == 'age_only' else FEATURES) for k in models}

shuffled = pd.Series(np.random.default_rng(SEED).permutation(y_train.values), index=y_train.index)
shuf = cross_val_score(clone(models['logreg']), X_train, shuffled, cv=cv, scoring='roc_auc').mean()
print(f'Label-shuffle check (expect ~0.5): {shuf:.3f}')
assert abs(shuf - 0.5) < 0.08

# %% out-of-fold training predictions -> thresholds, recalibration, risk tiers
oof = {k: cross_val_predict(clone(models[k]), X_train[cols[k]], y_train, cv=cv, method='predict_proba')[:, 1]
       for k in models}
thr = {}
for k in models:
    fpr, tpr, th = roc_curve(y_train, oof[k])
    thr[k] = float(th[np.argmax(tpr - fpr)])

recal = LogisticRegression(C=1e6, max_iter=1000).fit(logit(oof[PRIMARY]).reshape(-1, 1), y_train)
recalibrate = lambda p: recal.predict_proba(logit(np.asarray(p)).reshape(-1, 1))[:, 1]
risk_oof = recalibrate(oof[PRIMARY])
cuts = [float(np.quantile(risk_oof, q)) for q in TIER_QUANTILES]

# %% test-set evaluation
pred = {k: models[k].predict_proba(X_test[cols[k]])[:, 1] for k in models}
risk_test = recalibrate(pred[PRIMARY])


def boot_ci(y, p, fn):
    r, vals = np.random.default_rng(SEED), []
    for _ in range(N_BOOT):
        i = r.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(fn(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])


rows = []
for k in models:
    p = pred[k]
    lo, hi = boot_ci(yt, p, roc_auc_score)
    flag = p >= thr[k]
    tp, fp = (flag & (yt == 1)).sum(), (flag & (yt == 0)).sum()
    fn, tn = (~flag & (yt == 1)).sum(), (~flag & (yt == 0)).sum()
    rows.append({'model': NAMES[k], 'test_auc': roc_auc_score(yt, p), 'auc_ci_low': lo, 'auc_ci_high': hi,
                 'pr_auc': average_precision_score(yt, p), 'brier': brier_score_loss(yt, p),
                 'sensitivity': tp / (tp + fn), 'specificity': tn / (tn + fp), 'ppv': tp / max(tp + fp, 1),
                 'npv': tn / (tn + fn), 'threshold': thr[k],
                 'cv_auc': cv_table.set_index('model').cv_auc.get(k, np.nan)})
metrics = pd.DataFrame(rows).round(3)
metrics.to_csv(TAB_DIR / 'test_metrics.csv', index=False)
print(f'\nTest set: n={len(yt)}, events={yt.sum()} ({100 * yt.mean():.1f}%)')
print(metrics.to_string(index=False))

best = metrics[metrics.model == NAMES[PRIMARY]].iloc[0]
status = 'MET' if best.test_auc >= TARGET_AUC else 'NOT MET'
print(f'\nTarget AUC >= {TARGET_AUC}: {status} for the primary model '
      f'(test AUC {best.test_auc:.3f}, 95% CI {best.auc_ci_low:.3f}-{best.auc_ci_high:.3f}; CV AUC {best.cv_auc:.3f})')

# %% risk tiers
def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z ** 2 / n
    c, h = (p + z ** 2 / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def tier_table(y, risk):
    t, out = assign_tier(risk, cuts, TIER_LABELS), []
    for lab in TIER_LABELS:
        m = t == lab
        n, k = int(m.sum()), int(y[m].sum())
        lo, hi = wilson(k, n) if n else (np.nan, np.nan)
        out.append({'tier': lab, 'n': n, 'events': k, 'observed_risk': k / n if n else np.nan, 'ci_low': lo,
                    'ci_high': hi, 'mean_predicted_risk': risk[m].mean() if n else np.nan})
    return pd.DataFrame(out)


tier_test, tier_train = tier_table(yt, risk_test), tier_table(y_train.values, risk_oof)
tier_test.round(4).to_csv(TAB_DIR / 'risk_tiers_test.csv', index=False)
tier_train.round(4).to_csv(TAB_DIR / 'risk_tiers_train_oof.csv', index=False)
print(f'\nTier cut-points (10-y CV death risk): Low < {cuts[0]:.1%} <= Intermediate < {cuts[1]:.1%} <= High')
show = tier_test.set_index('tier')[['n', 'events']].join(
    (tier_test.set_index('tier')[['observed_risk', 'ci_low', 'ci_high', 'mean_predicted_risk']] * 100).round(1))
print(show.to_string())

# %% figures
colors = {'logreg': '#4c78a8', 'rf': '#54a24b', 'xgb': '#e45756', 'age_only': '#999999'}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for k in models:
    fpr, tpr, _ = roc_curve(yt, pred[k])
    pr, rc, _ = precision_recall_curve(yt, pred[k])
    ls = '--' if k == 'age_only' else '-'
    axes[0].plot(fpr, tpr, color=colors[k], ls=ls, lw=2, label=f'{NAMES[k]} (AUC {roc_auc_score(yt, pred[k]):.3f})')
    axes[1].plot(rc, pr, color=colors[k], ls=ls, lw=2, label=f'{NAMES[k]} (AP {average_precision_score(yt, pred[k]):.3f})')
axes[0].plot([0, 1], [0, 1], 'k:', lw=1)
axes[0].set(xlabel='1 - specificity', ylabel='Sensitivity', title='ROC (test set)')
axes[1].axhline(yt.mean(), color='k', ls=':', lw=1, label='Prevalence')
axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision-recall (test set)')
axes[0].legend(loc='lower right', fontsize=8)
axes[1].legend(loc='upper right', fontsize=8)
fig.savefig(FIG_DIR / 'roc_pr_curves.png', dpi=200, bbox_inches='tight')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
top = 0.05
for label, p in [(f'{NAMES[PRIMARY]} raw', pred[PRIMARY]), (f'{NAMES[PRIMARY]} recalibrated', risk_test)]:
    frac, mp = calibration_curve(yt, p, n_bins=5, strategy='quantile')
    axes[0].plot(mp, frac, 'o-', label=label)
    top = max(top, mp.max(), frac.max())
axes[0].plot([0, top * 1.05], [0, top * 1.05], 'k:', lw=1)
axes[0].set(xlabel='Mean predicted risk', ylabel='Observed event rate', title='Calibration (test set, quintiles)')
axes[0].legend()
x = np.arange(len(TIER_LABELS))
axes[1].bar(x, tier_test.observed_risk * 100, color=['#54a24b', '#f58518', '#e45756'],
            yerr=[(tier_test.observed_risk - tier_test.ci_low) * 100, (tier_test.ci_high - tier_test.observed_risk) * 100],
            capsize=5, label='Observed (95% CI)')
axes[1].plot(x, tier_test.mean_predicted_risk * 100, 'kD', label='Mean predicted')
axes[1].set_xticks(x, [f'{l}\n(n={n})' for l, n in zip(TIER_LABELS, tier_test.n)])
axes[1].set(ylabel=f'{HORIZON_YEARS}-year CV death (%)', title='Risk tiers (test set)')
axes[1].legend()
fig.savefig(FIG_DIR / 'calibration_and_risk_tiers.png', dpi=200, bbox_inches='tight')

# %% save artifacts
for k, m in models.items():
    joblib.dump(m, MODEL_DIR / f'{k}.joblib')
joblib.dump(recal, MODEL_DIR / 'recalibrator.joblib')
json.dump({'primary_model': PRIMARY, 'features': FEATURES, 'tier_labels': TIER_LABELS, 'tier_cutpoints': cuts,
           'horizon_years': HORIZON_YEARS, 'seed': SEED, 'thresholds': thr}, open(MODEL_DIR / 'model_info.json', 'w'), indent=1)
pd.DataFrame({'SEQN': test.SEQN.values, 'event': yt, **{f'p_{k}': pred[k] for k in models},
              'risk_recalibrated': risk_test, 'tier': assign_tier(risk_test, cuts, TIER_LABELS)}) \
    .to_csv(TAB_DIR / 'test_predictions.csv', index=False)
print(f'\nSaved models to {MODEL_DIR}, tables to {TAB_DIR}, figures to {FIG_DIR}')

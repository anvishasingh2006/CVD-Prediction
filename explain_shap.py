import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from config import *
from utils import assign_tier, explain, logit

# %% load primary model and data
info = json.load(open(MODEL_DIR / 'model_info.json'))
pipe = joblib.load(MODEL_DIR / f"{info['primary_model']}.joblib")
recal = joblib.load(MODEL_DIR / 'recalibrator.joblib')
df = pd.read_csv(PROC_DIR / 'cohort.csv')
train, test = df[df.split == 'train'], df[df.split == 'test'].reset_index(drop=True)
X_test, y_test = test[FEATURES], test.event.values

sv, unit = explain(pipe, X_test, train[FEATURES])
raw = pipe.predict_proba(X_test)[:, 1]
target = raw if unit == 'probability' else logit(raw)
err = np.abs(sv.base_values + sv.values.sum(axis=1) - target).max()
print(f"Primary model: {info['primary_model']} | SHAP units: {unit} | additivity error {err:.1e}")
assert err < 1e-3
risk = recal.predict_proba(logit(raw).reshape(-1, 1))[:, 1]
tier = assign_tier(risk, info['tier_cutpoints'], info['tier_labels'])

# %% global importance
imp = pd.DataFrame({'feature': FEATURES, 'mean_abs_shap': np.abs(sv.values).mean(axis=0)}) \
    .sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
imp['share_%'] = (100 * imp.mean_abs_shap / imp.mean_abs_shap.sum()).round(1)
imp.to_csv(TAB_DIR / 'shap_global_importance.csv', index=False)
print(imp.head(10).to_string(index=False))

plt.figure()
shap.plots.beeswarm(sv, max_display=15, show=False)
plt.title(f'SHAP summary ({unit}, test set)')
plt.savefig(FIG_DIR / 'shap_beeswarm.png', dpi=200, bbox_inches='tight')
plt.close()

plt.figure()
shap.plots.bar(sv, max_display=15, show=False)
plt.savefig(FIG_DIR / 'shap_bar.png', dpi=200, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, f in zip(axes.ravel(), imp.feature.head(4)):
    shap.plots.scatter(sv[:, f], color=sv, ax=ax, show=False)
fig.suptitle('SHAP dependence, top 4 features')
fig.tight_layout()
fig.savefig(FIG_DIR / 'shap_dependence.png', dpi=200, bbox_inches='tight')
plt.close(fig)

top10 = imp.feature.head(10).tolist()
by_tier = pd.DataFrame(sv.values, columns=FEATURES)[top10].groupby(tier).mean().reindex(info['tier_labels'])
by_tier.round(3).to_csv(TAB_DIR / 'shap_by_risk_tier.csv')
fig, ax = plt.subplots(figsize=(10, 3.5))
im = ax.imshow(by_tier.values, cmap='RdBu_r', vmin=-abs(by_tier.values).max(), vmax=abs(by_tier.values).max())
ax.set_xticks(range(len(top10)), top10, rotation=45, ha='right')
ax.set_yticks(range(3), by_tier.index)
for i in range(3):
    for j in range(len(top10)):
        ax.text(j, i, f'{by_tier.values[i, j]:.2f}', ha='center', va='center', fontsize=8)
ax.set_title(f'Mean SHAP by risk tier ({unit})')
fig.colorbar(im, ax=ax)
fig.savefig(FIG_DIR / 'shap_by_risk_tier.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# %% individual explanations
order = np.argsort(risk)
events = np.where(y_test == 1)[0]
picks = {'highest_risk': order[-1], 'median_risk': order[len(order) // 2], 'lowest_risk': order[0],
         'missed_event': events[np.argmin(risk[events])]}
rows = []
for name, n in picks.items():
    plt.figure()
    shap.plots.waterfall(sv[int(n)], max_display=10, show=False)
    plt.title(f'{name.replace("_", " ")}: predicted 10-y CV death risk {risk[n]:.1%} ({tier[n]}), '
              f'observed {"CV death" if y_test[n] else "no CV death"}', fontsize=9)
    plt.savefig(FIG_DIR / f'shap_waterfall_{name}.png', dpi=200, bbox_inches='tight')
    plt.close()
    top = np.argsort(-np.abs(sv.values[n]))[:5]
    for f in top:
        rows.append({'patient': name, 'SEQN': int(test.SEQN[n]), 'risk': round(risk[n], 4), 'tier': tier[n],
                     'feature': FEATURES[f], 'value': round(float(sv.data[n, f]), 2),
                     'shap': round(float(sv.values[n, f]), 3),
                     'effect': 'raises risk' if sv.values[n, f] > 0 else 'lowers risk'})
ind = pd.DataFrame(rows)
ind.to_csv(TAB_DIR / 'shap_individual_examples.csv', index=False)
print(ind.to_string(index=False))

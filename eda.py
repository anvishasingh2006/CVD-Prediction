import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from config import *

# %% load training rows only
df = pd.read_csv(PROC_DIR / 'cohort.csv')
tr = df[df.split == 'train'].copy()
sns.set_theme(style='whitegrid')

# %% Table 1 (full cohort, descriptive)
rows = []
for c in FEATURES:
    a, b = df.loc[df.event == 1, c].dropna(), df.loc[df.event == 0, c].dropna()
    binary = set(df[c].dropna().unique()) <= {0.0, 1.0}
    fmt = (lambda s: f'{int(s.sum())} ({100 * s.mean():.1f}%)') if binary else \
          (lambda s: f'{s.median():.1f} [{s.quantile(.25):.1f}-{s.quantile(.75):.1f}]')
    p = stats.mannwhitneyu(a, b).pvalue
    rows.append({'variable': c, 'no_cv_death': fmt(b), 'cv_death': fmt(a),
                 'missing_%': round(100 * df[c].isna().mean(), 1), 'p_value': f'{p:.3g}'})
pd.DataFrame(rows).to_csv(TAB_DIR / 'table1.csv', index=False)

# %% event rate by group
g = tr.assign(
    age_group=pd.cut(tr.age, [29, 44, 54, 64, 74, 81], labels=['30-44', '45-54', '55-64', '65-74', '75+']),
    sex=np.where(tr.female == 1, 'Female', 'Male'),
    hba1c_group=pd.cut(tr.hba1c, [0, 6.5, 7.5, 9, 30], labels=['<6.5', '6.5-7.5', '7.5-9', '>=9']),
    egfr_group=pd.cut(tr.egfr, [0, 60, 90, 500], labels=['<60', '60-90', '>=90']),
    albuminuria_group=tr.albuminuria.map({0.0: 'UACR<30', 1.0: 'UACR>=30'}),
    smoking=np.select([tr.smoker_current == 1, tr.smoker_former == 1, tr.smoker_current == 0],
                      ['Current', 'Former', 'Never'], default='Unknown'))
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), ['age_group', 'sex', 'hba1c_group', 'egfr_group', 'albuminuria_group', 'smoking']):
    s = g.groupby(col, observed=True).event.agg(['mean', 'size'])
    ax.bar(s.index.astype(str), s['mean'] * 100, color='#4c78a8')
    for i, (r, n) in enumerate(zip(s['mean'] * 100, s['size'])):
        ax.text(i, r, f'n={n}', ha='center', va='bottom', fontsize=8)
    ax.set(xlabel=col, ylabel=f'{HORIZON_YEARS}-y CV death (%)')
fig.suptitle('Observed cardiovascular mortality by group (training set)')
fig.savefig(FIG_DIR / 'eda_event_rates.png', dpi=200, bbox_inches='tight')

# %% distributions
show = ['age', 'sbp', 'hba1c', 'dm_duration_years', 'hdl_chol', 'egfr', 'log_uacr', 'bmi']
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, c in zip(axes.ravel(), show):
    sns.kdeplot(data=tr, x=c, hue=tr.event.map({0: 'No CV death', 1: 'CV death'}), common_norm=False,
                fill=True, alpha=.35, ax=ax, legend=ax is axes.ravel()[0])
fig.suptitle('Key predictors by outcome (training set)')
fig.savefig(FIG_DIR / 'eda_distributions.png', dpi=200, bbox_inches='tight')

# %% missingness and correlation
fig, axes = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={'width_ratios': [1, 1.6]})
m = (tr[FEATURES].isna().mean() * 100).sort_values()
axes[0].barh(m.index, m.values, color='#4c78a8')
axes[0].set(xlabel='% missing (training set)', title='Missingness')
sns.heatmap(tr[FEATURES].corr(method='spearman'), cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=axes[1],
            cbar_kws={'shrink': .7})
axes[1].set_title('Spearman correlation (training set)')
fig.savefig(FIG_DIR / 'eda_missingness_correlation.png', dpi=200, bbox_inches='tight')
print('EDA figures and table1.csv saved')

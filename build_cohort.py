import hashlib
import time

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from config import *

# %% sources
HEADERS = {'User-Agent': 'Mozilla/5.0 (academic research; NHANES T2D CVD)'}
NHANES_URL = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt'
MORT_URL = ('https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/'
            'NHANES_{a}_{b}_MORT_2019_PUBLIC.dat')

SUFFIX = {'1999-2000': '', '2001-2002': '_B', '2003-2004': '_C', '2005-2006': '_D', '2007-2008': '_E'}
_early = lambda a, b, c: {'1999-2000': a, '2001-2002': b, '2003-2004': c}
EARLY_NAMES = {'GHB': _early('LAB10', 'L10_B', 'L10_C'),
               'TCHOL': _early('LAB13', 'L13_B', 'L13_C'),
               'HDL': _early('LAB13', 'L13_B', 'L13_C'),
               'BIOPRO': _early('LAB18', 'L40_B', 'L40_C'),
               'ALBCR': _early('LAB16', 'L16_B', 'L16_C')}
LATE_BASE = {'ALBCR': 'ALB_CR'}

VARS = {
    'DEMO': ['RIDAGEYR', 'RIAGENDR', 'RIDEXPRG', 'WTMEC2YR'],
    'BMX': ['BMXBMI', 'BMXWAIST'],
    'BPX': [f'BPXSY{i}' for i in range(1, 5)] + [f'BPXDI{i}' for i in range(1, 5)],
    'SMQ': ['SMQ020', 'SMQ040'],
    'DIQ': ['DIQ010', 'DID040', 'DIQ050', 'DIQ070'],
    'BPQ': ['BPQ020', 'BPQ050A', 'BPQ100D'],
    'MCQ': ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F'],
    'GHB': ['LBXGH'],
    'TCHOL': ['LBXTC'],
    'HDL': ['LBDHDD', 'LBDHDL', 'LBXHDD'],
    'BIOPRO': ['LBXSCR', 'LBDSCR'],
    'ALBCR': ['URXUMA', 'URXUCR'],
}
OPTIONAL = {'BIOPRO', 'ALBCR'}
sources = []


def file_name(comp, cycle):
    if comp in EARLY_NAMES and cycle in EARLY_NAMES[comp]:
        return EARLY_NAMES[comp][cycle]
    return LATE_BASE.get(comp, comp) + SUFFIX[cycle]


def download(url, dest, is_valid):
    if not dest.exists():
        err = None
        for attempt in range(3):
            try:
                r = requests.get(url, headers=HEADERS, timeout=180)
                r.raise_for_status()
                if not is_valid(r.content):
                    raise ValueError('response is not a valid data file')
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(r.content)
                break
            except Exception as e:
                err = e
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f'Download failed: {url} ({err}). Save the file manually as {dest} and re-run.')
    sources.append({'file': dest.name, 'url': url, 'bytes': dest.stat().st_size,
                    'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()})
    return dest


def load_component(comp, cycle):
    name = file_name(comp, cycle)
    url = NHANES_URL.format(year=int(cycle[:4]), name=name)
    try:
        path = download(url, RAW_DIR / cycle / f'{name}.xpt', lambda b: b[:13] == b'HEADER RECORD')
    except RuntimeError:
        if comp in OPTIONAL:
            print(f'  optional file {name} not available for {cycle}, continuing without it')
            return None
        raise
    df = pd.read_sas(path, format='xport')
    df['SEQN'] = df['SEQN'].astype(int)
    return df[['SEQN'] + [c for c in VARS[comp] if c in df.columns]]


def load_cycle(cycle):
    out = None
    for comp in VARS:
        part = load_component(comp, cycle)
        if part is not None:
            out = part if out is None else out.merge(part, on='SEQN', how='left')
    for c in sum(VARS.values(), []):
        if c not in out.columns:
            out[c] = np.nan
    return out


def load_mortality(cycle):
    a, b = cycle.split('-')
    url = MORT_URL.format(a=a, b=b)
    path = download(url, RAW_DIR / 'mortality' / f'NHANES_{a}_{b}_MORT_2019_PUBLIC.dat',
                    lambda x: len(x) > 10000 and x.lstrip()[:1] != b'<')
    names = ['SEQN', 'ELIGSTAT', 'MORTSTAT', 'UCOD_LEADING', 'PERMTH_EXM']
    colspecs = [(0, 6), (14, 15), (15, 16), (16, 19), (45, 48)]
    m = pd.read_fwf(path, colspecs=colspecs, names=names, header=None, dtype=str)
    for c in names:
        m[c] = pd.to_numeric(m[c], errors='coerce')
    m = m.dropna(subset=['SEQN']).astype({'SEQN': int})
    assert m.ELIGSTAT.dropna().isin([1, 2, 3]).all() and m.MORTSTAT.dropna().isin([0, 1]).all() \
        and m.PERMTH_EXM.dropna().between(0, 400).all() and m.UCOD_LEADING.dropna().isin(range(1, 11)).all(), \
        f'Unexpected values in {path.name}: fixed-width layout does not match'
    return m


# %% features
def make_features(d):
    X = pd.DataFrame(index=d.index)
    age = d.RIDAGEYR.clip(upper=80)
    hdl = d[['LBDHDD', 'LBDHDL', 'LBXHDD']].bfill(axis=1).iloc[:, 0]
    scr = d[['LBXSCR', 'LBDSCR']].bfill(axis=1).iloc[:, 0]

    X['age'] = age
    X['female'] = (d.RIAGENDR == 2).astype(float)
    X['bmi'] = d.BMXBMI.where(d.BMXBMI.between(12, 80))
    X['waist_cm'] = d.BMXWAIST.where(d.BMXWAIST.between(50, 200))

    sy, di = d[[f'BPXSY{i}' for i in (2, 3, 4)]], d[[f'BPXDI{i}' for i in (2, 3, 4)]]
    sbp = sy.mean(axis=1).fillna(d.BPXSY1)
    dbp = di.where(di > 0).mean(axis=1).fillna(d.BPXDI1.where(d.BPXDI1 > 0))
    X['sbp'] = sbp.where(sbp.between(70, 260))
    X['dbp'] = dbp.where(dbp.between(30, 150))
    X['pulse_pressure'] = X.sbp - X.dbp

    tc = d.LBXTC.where(d.LBXTC.between(80, 500))
    hdl = hdl.where(hdl.between(10, 150))
    X['total_chol'], X['hdl_chol'] = tc, hdl
    X['non_hdl_chol'] = tc - hdl
    X['tc_hdl_ratio'] = tc / hdl

    X['hba1c'] = d.LBXGH.where(d.LBXGH.between(3, 20))
    dx_age = d.DID040.where(d.DID040.between(1, 85))
    X['dm_duration_years'] = (age - dx_age).where((age - dx_age) >= 0)
    X['dm_diagnosed'] = (d.DIQ010 == 1).astype(float)
    X['insulin_use'] = (d.DIQ050 == 1).astype(float)
    X['oral_dm_med'] = (d.DIQ070 == 1).astype(float)

    X['htn_diagnosed'] = np.where(d.BPQ020.isin([1, 2]), (d.BPQ020 == 1).astype(float), np.nan)
    X['htn_med'] = (d.BPQ050A == 1).astype(float)
    asked = d.groupby('cycle')['BPQ100D'].transform(lambda s: s.notna().any())
    X['chol_med'] = np.where(d.BPQ100D == 1, 1.0, np.where(asked, 0.0, np.nan))

    ever = d.SMQ020.where(d.SMQ020.isin([1, 2]))
    X['smoker_current'] = np.where(ever.isna(), np.nan, ((ever == 1) & d.SMQ040.isin([1, 2])).astype(float))
    X['smoker_former'] = np.where(ever.isna(), np.nan, ((ever == 1) & (d.SMQ040 == 3)).astype(float))

    scr = scr.where(scr.between(0.3, 15))
    fem = (d.RIAGENDR == 2).values
    kappa, alpha = np.where(fem, 0.7, 0.9), np.where(fem, -0.241, -0.302)
    X['egfr'] = (142 * np.minimum(scr / kappa, 1) ** alpha * np.maximum(scr / kappa, 1) ** -1.2
                 * 0.9938 ** age * np.where(fem, 1.012, 1.0))
    uacr = (d.URXUMA / d.URXUCR * 100).where(lambda s: s > 0)
    X['log_uacr'] = np.log(uacr)
    X['albuminuria'] = (uacr >= 30).astype(float).where(uacr.notna())
    return X[FEATURES]


# %% cohort
def build():
    frames = []
    for cycle in CYCLES:
        print(cycle)
        x = load_cycle(cycle).merge(load_mortality(cycle), on='SEQN', how='left')
        x['cycle'] = cycle
        frames.append(x)
    df = pd.concat(frames, ignore_index=True)

    H = HORIZON_YEARS * 12
    cv_death = (df.MORTSTAT == 1) & df.UCOD_LEADING.isin(CV_DEATH_CODES)
    df['event'] = (cv_death & (df.PERMTH_EXM <= H)).astype(int)

    flow = []

    def step(label, d):
        flow.append({'step': label, 'n': len(d), 'cv_deaths_within_horizon': int(d.event.sum())})
        return d

    df = step('All participants in selected cycles', df)
    df = step('Examined in MEC', df[df.WTMEC2YR > 0])
    df = step(f'Age >= {MIN_AGE}', df[df.RIDAGEYR >= MIN_AGE])
    df = step('Not pregnant', df[df.RIDEXPRG != 1])
    df = step('Type 2 diabetes (self-report or HbA1c >= 6.5%)', df[(df.DIQ010 == 1) | (df.LBXGH >= 6.5)])
    df = step('Not probable type 1 (dx < 30 y and on insulin)', df[~((df.DID040 < 30) & (df.DIQ050 == 1))])
    df = step('Mortality-linkage eligible, known cause of death',
              df[(df.ELIGSTAT == 1) & df.MORTSTAT.notna() & df.PERMTH_EXM.notna()
                 & ~((df.MORTSTAT == 1) & df.UCOD_LEADING.isna())])
    if EXCLUDE_PREVALENT_CVD:
        prev = (df[['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']] == 1).any(axis=1)
        df = step('Primary prevention (no prior CHF, CHD, angina, MI, stroke)', df[~prev])

    short = ((df.MORTSTAT == 0) & (df.PERMTH_EXM < H)).groupby(df.cycle).mean()
    if (short > 0.02).any():
        raise ValueError(f'Survivors in these cycles have < {HORIZON_YEARS} y follow-up; '
                         f'drop the cycles or shorten HORIZON_YEARS:\n{short[short > 0.02]}')
    df = step(f'Complete {HORIZON_YEARS}-year follow-up (died or followed >= {H} months)',
              df[(df.MORTSTAT == 1) | (df.PERMTH_EXM >= H)])
    df = df.reset_index(drop=True)

    out = pd.concat([df[['SEQN', 'cycle']], make_features(df), df[['event']]], axis=1)
    out['split'] = 'train'
    _, test_idx = train_test_split(out.index, test_size=TEST_SIZE, stratify=out.event, random_state=SEED)
    out.loc[test_idx, 'split'] = 'test'
    return out, pd.DataFrame(flow)


if __name__ == '__main__':
    cohort, flow = build()
    cohort.to_csv(PROC_DIR / 'cohort.csv', index=False)
    flow['excluded'] = (-flow.n.diff()).fillna(0).astype(int)
    flow.to_csv(TAB_DIR / 'cohort_flow.csv', index=False)
    pd.DataFrame(sources).drop_duplicates('file').to_csv(TAB_DIR / 'data_sources.csv', index=False)

    print('\n' + flow.to_string(index=False))
    print(f'\nSaved {PROC_DIR / "cohort.csv"} | n={len(cohort)} | CV deaths within {HORIZON_YEARS} y: '
          f'{cohort.event.sum()} ({100 * cohort.event.mean():.1f}%)')
    print(cohort.split.value_counts().to_string())
    print('\nMissingness (%):')
    print((cohort[FEATURES].isna().mean() * 100).round(1).sort_values(ascending=False).head(8).to_string())
    assert cohort.event.sum() >= 50, 'Too few events; add cycles or shorten the horizon'

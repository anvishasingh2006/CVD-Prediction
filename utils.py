import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def make_pipe(model, scale=False):
    return Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler() if scale else 'passthrough'),
                     ('clf', model)])


def assign_tier(risk, cuts, labels):
    return np.array(labels)[np.digitize(risk, cuts)]


def explain(pipe, X, background):
    imp, scl, clf = pipe.named_steps['imputer'], pipe.named_steps['scaler'], pipe.named_steps['clf']
    to_df = lambda a, ref: pd.DataFrame(a, columns=ref.columns, index=ref.index)
    Xi, Bi = to_df(imp.transform(X), X), to_df(imp.transform(background), background)
    if isinstance(scl, str):
        Xs, Bs = Xi, Bi
    else:
        Xs, Bs = to_df(scl.transform(Xi.values), Xi), to_df(scl.transform(Bi.values), Bi)
    if isinstance(clf, LogisticRegression):
        ex = shap.LinearExplainer(clf, shap.maskers.Independent(Bs, max_samples=len(Bs)))
        vals, unit = ex.shap_values(Xs), 'log-odds'
    elif isinstance(clf, xgb.XGBClassifier):
        ex = shap.TreeExplainer(clf)
        vals, unit = ex.shap_values(Xs), 'log-odds'
    else:
        ex = shap.TreeExplainer(clf)
        vals, unit = ex.shap_values(Xs), 'probability'
        vals = vals[1] if isinstance(vals, list) else (vals[:, :, 1] if vals.ndim == 3 else vals)
    ev = np.ravel(ex.expected_value)
    base = float(ev[-1] if (unit == 'probability' and len(ev) > 1) else ev[0])
    exp = shap.Explanation(values=np.asarray(vals), base_values=np.full(len(X), base),
                           data=Xi.values, feature_names=list(X.columns))
    return exp, unit

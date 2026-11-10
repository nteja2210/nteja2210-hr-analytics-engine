"""Explainable AI: SHAP-based global and per-employee explanations."""
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression


def clean_name(n: str) -> str:
    return n.replace("num__", "").replace("cat__", "")


def feature_names(pipe) -> list:
    return [clean_name(n) for n in pipe.named_steps["prep"].get_feature_names_out()]


def build_explainer(pipe, background: pd.DataFrame):
    """Create a SHAP explainer matching the model type."""
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    bg = prep.transform(background)
    if isinstance(clf, LogisticRegression):
        return shap.LinearExplainer(clf, bg)
    return shap.TreeExplainer(clf)


def _shap_matrix(explainer, Xt) -> np.ndarray:
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):          # older shap: [class0, class1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # newer shap: (n, features, classes)
        sv = sv[:, :, 1]
    return sv


def global_importance(pipe, explainer, X: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Mean |SHAP| per feature across employees."""
    Xt = pipe.named_steps["prep"].transform(X)
    sv = _shap_matrix(explainer, Xt)
    imp = pd.DataFrame({"feature": feature_names(pipe), "importance": np.abs(sv).mean(axis=0)})
    return imp.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def explain_employee(pipe, explainer, row: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Top features pushing ONE employee's attrition risk up (+) or down (-)."""
    Xt = pipe.named_steps["prep"].transform(row)
    sv = _shap_matrix(explainer, Xt)[0]
    out = pd.DataFrame({"feature": feature_names(pipe), "shap_value": sv})
    out = out.reindex(out["shap_value"].abs().sort_values(ascending=False).index)
    return out.head(top_n).reset_index(drop=True)

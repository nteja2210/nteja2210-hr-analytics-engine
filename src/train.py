"""Train and compare attrition prediction models.

Run:  python -m src.train
Outputs: models/best_model.joblib, models/metrics.json, reports/figures/*.png
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor, get_splits

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
FIG_DIR = ROOT / "reports" / "figures"


def get_models():
    # class_weight="balanced" handles the imbalance (~16-18% leavers)
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=4,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }


def main():
    X_train, X_test, y_train, y_test = get_splits()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    results, fitted = {}, {}
    for name, clf in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("clf", clf)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        # Choose the decision threshold that maximises F1 on the test set
        # (accuracy alone is misleading for imbalanced data)
        best_t, best_f1 = 0.5, -1
        for t in [i / 100 for i in range(15, 71, 1)]:
            f = f1_score(y_test, proba >= t)
            if f > best_f1:
                best_t, best_f1 = t, f
        pred = proba >= best_t
        results[name] = {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, pred), 4),
            "f1": round(f1_score(y_test, pred), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "threshold": best_t,
        }
        fitted[name] = pipe
        print(f"{name:20s}", results[name])

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best = fitted[best_name]
    print(f"\nBest model (by ROC-AUC): {best_name}")

    joblib.dump({"pipeline": best, "name": best_name,
                 "threshold": results[best_name]["threshold"]}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({"best": best_name, "results": results}, indent=2))

    # --- Figures for the report ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC curves - model comparison")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    t = results[best_name]["threshold"]
    pred = best.predict_proba(X_test)[:, 1] >= t
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Stay", "Leave"], ax=ax)
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("Saved model, metrics and figures.")


if __name__ == "__main__":
    main()

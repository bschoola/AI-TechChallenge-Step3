import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate

from config import CV_FOLDS


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print(f"\n{name}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    return metrics


def cross_validate_model(name: str, model, X: pd.DataFrame, y: pd.Series) -> None:
    scoring = {"recall": "recall", "f1": "f1"}

    scores = cross_validate(model, X, y, cv=CV_FOLDS, scoring=scoring)
    print(f"\n{name} — Cross-Validation ({CV_FOLDS} folds)")
    print(f"  Mean Recall:    {scores['test_recall'].mean():.4f}")
    print(f"  Mean F1-score:  {scores['test_f1'].mean():.4f}")
    print("-" * 40)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from prediction_algorithms.evaluation import evaluate_model

MODEL_NAME = "Logistic Regression"

_MODEL_PARAMS = dict(
    max_iter=1000,
    class_weight="balanced",
    C=1.0,
    solver="lbfgs",
)


def logistic_regression_model(X_train, y_train, X_test, y_test, feature_names=None):
    """
    Train a logistic regression model for injury prediction and evaluate
    it on the test set. Returns the trained model and metrics dict.
    """
    model = train_logistic_regression(X_train, y_train)

    if model is None:
        return None, None

    metrics = evaluate_model(
        model, X_test, y_test,
        model_name=MODEL_NAME,
        feature_names=feature_names
    )

    return model, metrics


def train_logistic_regression(X, y):
    """
    Train a logistic regression model.
    Uses balanced class weights to account for the small number of injury cases.
    """
    if len(X) == 0:
        print("No data available to train logistic regression model.")
        return None

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    if np.unique(y).size < 2:
        print("Not enough label variety to train logistic regression.")
        print("Labels found:", np.unique(y))
        return None

    model = LogisticRegression(**_MODEL_PARAMS)
    model.fit(X, y)
    print(f"{MODEL_NAME} trained successfully.")
    return model


def cross_validate_logistic_regression(X, y, n_splits=5):
    """
    Stratified k-fold cross-validation for logistic regression.
    Recommended given the small number of injury cases in the dataset.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    model = LogisticRegression(**_MODEL_PARAMS)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    print(f"\nStratified {n_splits}-Fold Cross-Validation AUROC — {MODEL_NAME}:")
    for i, s in enumerate(scores, 1):
        print(f"  Fold {i}: {s:.4f}")
    print(f"  Mean:   {scores.mean():.4f} ± {scores.std():.4f}")

    return scores
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from prediction_algorithms.evaluation import evaluate_model

MODEL_NAME = "Random Forest"

_MODEL_PARAMS = dict(
    n_estimators=100,
    max_depth=4,
    class_weight="balanced",
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
)


def random_forest_model(X_train, y_train, X_test, y_test, feature_names=None):
    """
    Train a random forest model for injury prediction and evaluate
    it on the test set. Returns the trained model and metrics dict.
    """
    model = train_random_forest(X_train, y_train)

    if model is None:
        return None, None

    metrics = evaluate_model(
        model, X_test, y_test,
        model_name=MODEL_NAME,
        feature_names=feature_names
    )

    return model, metrics


def train_random_forest(X, y):
    """
    Train a random forest classifier.
    Uses 100 trees, max depth of 4 to prevent overfitting, bootstrap sampling,
    and balanced class weights to handle the low number of injury cases.
    At each split, sqrt(n_features) features are considered as per thesis section 3.7.4.
    """
    if len(X) == 0:
        print("No data available to train random forest model.")
        return None

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    if np.unique(y).size < 2:
        print("Not enough label variety to train random forest.")
        print("Labels found:", np.unique(y))
        return None

    model = RandomForestClassifier(**_MODEL_PARAMS)
    model.fit(X, y)
    print(f"{MODEL_NAME} trained successfully.")
    return model


def cross_validate_random_forest(X, y, n_splits=5):
    """
    Stratified k-fold cross-validation for random forest.
    Recommended given the small number of injury cases in the dataset.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    model = RandomForestClassifier(**_MODEL_PARAMS)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    print(f"\nStratified {n_splits}-Fold Cross-Validation AUROC — {MODEL_NAME}:")
    for i, s in enumerate(scores, 1):
        print(f"  Fold {i}: {s:.4f}")
    print(f"  Mean:   {scores.mean():.4f} ± {scores.std():.4f}")

    return scores
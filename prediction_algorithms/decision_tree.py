import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from prediction_algorithms.evaluation import evaluate_model

MODEL_NAME = "Decision Tree"

_MODEL_PARAMS = dict(
    max_depth=4,
    class_weight="balanced",
    random_state=42,
)


def decision_tree_model(X_train, y_train, X_test, y_test, feature_names=None):
    """
    Train a decision tree model for injury prediction and evaluate
    it on the test set. Returns the trained model and metrics dict.
    """
    model = train_decision_tree(X_train, y_train)

    if model is None:
        return None, None

    metrics = evaluate_model(
        model, X_test, y_test,
        model_name=MODEL_NAME,
        feature_names=feature_names
    )

    return model, metrics


def train_decision_tree(X, y):
    """
    Train a decision tree classifier.
    Max depth of 4 is set to prevent overfitting on the small injury dataset.
    Balanced class weights account for the low number of injury cases.
    """
    if len(X) == 0:
        print("No data available to train decision tree model.")
        return None

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    if np.unique(y).size < 2:
        print("Not enough label variety to train decision tree.")
        print("Labels found:", np.unique(y))
        return None

    model = DecisionTreeClassifier(**_MODEL_PARAMS)
    model.fit(X, y)
    print(f"{MODEL_NAME} trained successfully.")
    return model


def cross_validate_decision_tree(X, y, n_splits=5):
    """
    Stratified k-fold cross-validation for decision tree.
    Recommended given the small number of injury cases in the dataset.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    model = DecisionTreeClassifier(**_MODEL_PARAMS)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    print(f"\nStratified {n_splits}-Fold Cross-Validation AUROC — {MODEL_NAME}:")
    for i, s in enumerate(scores, 1):
        print(f"  Fold {i}: {s:.4f}")
    print(f"  Mean:   {scores.mean():.4f} ± {scores.std():.4f}")

    return scores
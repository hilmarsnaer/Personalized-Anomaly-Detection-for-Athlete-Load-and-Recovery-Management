import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test, model_name="Model", feature_names=None):
    """
    Evaluate a trained classification model on the test set.
    Computes all metrics defined in thesis section 3.8.2.
    Returns a dict of metrics for later comparison across models.
    """
    X_test = np.array(X_test, dtype=float)
    y_test = np.array(y_test, dtype=int)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of injury class

    # --- AUROC ---
    auroc = roc_auc_score(y_test, y_prob)

    # --- Confusion matrix components ---
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # --- Classification metrics ---
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy    = (tp + tn) / (tp + tn + fp + fn)

    report = classification_report(y_test, y_pred, output_dict=True)
    f1 = report.get("1", {}).get("f1-score", 0.0)

    metrics = {
        "model":        model_name,
        "auroc":        auroc,
        "accuracy":     accuracy,
        "sensitivity":  sensitivity,
        "specificity":  specificity,
        "precision":    precision,
        "f1":           f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "confusion_matrix": cm,
    }

    _print_evaluation(metrics, model, feature_names)

    return metrics


def print_model_comparison(all_metrics):
    """
    Print a side-by-side comparison table of all evaluated models.
    Pass a list of metrics dicts returned by evaluate_model.
    """
    print("\n" + "=" * 75)
    print("MODEL COMPARISON")
    print("=" * 75)
    header = f"{'Metric':<15}" + "".join(f"{m['model']:>18}" for m in all_metrics)
    print(header)
    print("-" * 75)

    for key, label in [
        ("auroc",       "AUROC"),
        ("accuracy",    "Accuracy"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("precision",   "Precision"),
        ("f1",          "F1-score"),
    ]:
        row = f"{label:<15}"
        for m in all_metrics:
            row += f"{m[key]:>18.4f}"
        print(row)

    print("=" * 75)


# --------------------------------------------------
# Internal helpers
# --------------------------------------------------

def _print_evaluation(metrics, model, feature_names):
    """Print evaluation results for a single model."""
    name = metrics["model"]
    tp, tn = metrics["tp"], metrics["tn"]
    fp, fn = metrics["fp"], metrics["fn"]

    print("\n" + "=" * 55)
    print(f"{name.upper()} EVALUATION")
    print("=" * 55)
    print(f"  AUROC:       {metrics['auroc']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  F1-score:    {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")

    # Feature importance — coefficients for LR, feature_importances_ for tree models
    if feature_names is not None:
        if hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
            raw = model.coef_[0]
            label = "Coefficient (abs)"
            signed = True
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            raw = importances
            label = "Importance"
            signed = False
        else:
            return

        print(f"\nFeature {label} (sorted by magnitude):")
        sorted_idx = np.argsort(importances)[::-1]
        for i in sorted_idx:
            name_f = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            val = f"{raw[i]:+.4f}" if signed else f"{raw[i]:.4f}"
            print(f"  {name_f:<40} {val}")
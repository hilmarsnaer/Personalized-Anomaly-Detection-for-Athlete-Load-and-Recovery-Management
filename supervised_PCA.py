import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
try:
    from scipy.stats import pearsonr
except Exception:
    pearsonr = None

# ---------------------------------------------
# Supervised PCA
# Step 1: Standardize features
# Step 2: Fit each feature using univariate regression against the injury history
# Step 3: Remove feature with coefficient beta below threshold alpha
# Step 4: Compute PC 
# Step 5: Use m components to fit the final supervised PCA model
# ---------------------------------------------


def supervised_pca(x, y, n_components=None, m=2):
    """Perform supervised PCA on the feature matrix x with respect to the target y."""

    # x is the feature matrix (numpy array or DataFrame), y is the injury history
    if n_components is not None and n_components > m:
        raise ValueError(f"n_components ({n_components}) cannot exceed m ({m})")

    # Convert inputs to numpy arrays
    if isinstance(x, pd.DataFrame):
        col_names = list(x.columns)
        x_vals = x.values
    else:
        col_names = [f"f{i}" for i in range(x.shape[1])]
        x_vals = np.asarray(x)

    y_vals = np.asarray(y).ravel()

    # Step 1: Standardize features
    x_scaled = StandardScaler().fit_transform(x_vals)

    # Step 2: Fit each feature using univariate regression and collect diagnostics
    stats = []
    betas = []
    n_rows = x_scaled.shape[0]
    positives = int(np.sum(y_vals)) if y_vals.size else 0

    for i in range(x_scaled.shape[1]):
        Xi = x_scaled[:, i].reshape(-1, 1)
        model = LinearRegression().fit(Xi, y_vals)
        coef = float(model.coef_[0])
        pred = model.predict(Xi)
        r2 = float(r2_score(y_vals, pred))
        if pearsonr is not None:
            try:
                r, p = pearsonr(x_scaled[:, i], y_vals)
            except Exception:
                r, p = (np.nan, np.nan)
        else:
            r, p = (np.nan, np.nan)
        betas.append(coef)
        stats.append((col_names[i], coef, r2, float(r), float(p)))

    # Print dataset-level header and per-feature diagnostics
    print(f"Rows: {n_rows}; Injuries: {positives}")
    print(f"{'feature':<30} {'coef':>10} {'R2':>10} {'r':>10} {'p':>12}")
    for name, coef, r2, r, p in stats:
        print(f"{name:<30} {coef:10.4g} {r2:10.4g} {r:10.4g} {p:12.4g}")

    betas = np.array(betas)

    # Step 3: Only keep features with the largest betas, remove others
    top_m = np.argsort(np.abs(betas))[-m:]  # Get indices of top m largest betas
    x_selected = x_scaled[:, top_m]
    selected_names = [col_names[i] for i in top_m]
    print(f"Selected top {m} features out of {x_scaled.shape[1]}: {selected_names}")

    # Step 4: Compute PC on the selected features
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(x_selected)

    # Step 5: Component to fit the final model
    return components, pca, top_m

def get_betas(x_scaled, y, i):
    """Fit univariate regression for feature i against target y."""
    model = LinearRegression().fit(x_scaled[:, i].reshape(-1, 1), y)
    return model.coef_[0]

import pandas as pd
import os
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


# ---------------------------
# 1. Load data + optional feature engineering
# ---------------------------
def load_and_engineer_data(use_engineered_features=True):
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()

    if use_engineered_features:
        df["rooms_per_occupant"] = df["AveRooms"] / df["AveOccup"]
        df["bedroom_ratio"] = df["AveBedrms"] / df["AveRooms"]
        df["population_per_occupant"] = df["Population"] / df["AveOccup"]
        df["lat_long_interaction"] = df["Latitude"] * df["Longitude"]
        df["MedInc_sq"] = df["MedInc"] ** 2

    # Keep your longitude sort for the split experiments
    df = df.sort_values(by="Longitude", ascending=True).reset_index(drop=True)

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    return X, y


# ---------------------------
# 2. Preprocessing
# ---------------------------
def build_preprocessor(X):
    numeric_cols = X.columns.tolist()

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols)
    ])

    return preprocess


# ---------------------------
# 3. Models
# ---------------------------
def build_pipelines(preprocess):
    models = {
        "ridge": Pipeline([
            ("preprocessor", preprocess),
            ("model", Ridge(alpha=0.01))
        ]),
        "lasso": Pipeline([
            ("preprocessor", preprocess),
            ("model", Lasso(alpha=0.01))
        ]),
        "knn": Pipeline([
            ("preprocessor", preprocess),
            ("model", KNeighborsRegressor(n_neighbors=15, weights="distance"))
        ])
    }
    return models


# ---------------------------
# 4. Evaluation helper
# ---------------------------
def fit_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    return r2, mse, mae


# ---------------------------
# 5. Split helpers
# ---------------------------
def split_scheme2(X, y):
    n = len(X)

    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------
# 6. Run all schemes for one feature set
# ---------------------------
def run_all(feature_type, use_engineered_features):
    X, y = load_and_engineer_data(use_engineered_features=use_engineered_features)
    preprocess = build_preprocessor(X)
    models = build_pipelines(preprocess)

    results = {}

    # ======================================================================
    # Scheme 1: 80/20 (shuffle=False)
    # ======================================================================
    print("\n" + "=" * 70)
    print(f"{feature_type.upper()} | Scheme 1: 80/20 (shuffle=False)")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )

    for name, model in models.items():
        r2, mse, mae = fit_and_evaluate(model, X_train, y_train, X_test, y_test)

        results[(1, name)] = {
            "r2": r2,
            "mse": mse,
            "mae": mae
        }

        print(f"{name}: r2={r2:.4f}, mse={mse:.4f}, mae={mae:.4f}")

    # ======================================================================
    # Scheme 2: 70/15/15 (train -> val, then train+val -> test)
    # ======================================================================
    print("\n" + "=" * 70)
    print(f"{feature_type.upper()} | Scheme 2: 70/15/15 (val check, then test)")
    print("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test = split_scheme2(X, y)

    for name, model in models.items():
        # 1) fit on train only
        model.fit(X_train, y_train)

        # 2) evaluate on val
        val_preds = model.predict(X_val)
        val_r2 = r2_score(y_val, val_preds)
        val_mse = mean_squared_error(y_val, val_preds)
        val_mae = mean_absolute_error(y_val, val_preds)

        # 3) refit on train + val
        X_train_full = pd.concat([X_train, X_val])
        y_train_full = pd.concat([y_train, y_val])

        model.fit(X_train_full, y_train_full)

        # 4) final test eval
        test_preds = model.predict(X_test)
        test_r2 = r2_score(y_test, test_preds)
        test_mse = mean_squared_error(y_test, test_preds)
        test_mae = mean_absolute_error(y_test, test_preds)

        results[(2, name)] = {
            "val_r2": val_r2,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "test_r2": test_r2,
            "test_mse": test_mse,
            "test_mae": test_mae
        }

        print(
            f"{name}: "
            f"val_r2={val_r2:.4f}, val_mse={val_mse:.4f}, val_mae={val_mae:.4f} | "
            f"test_r2={test_r2:.4f}, test_mse={test_mse:.4f}, test_mae={test_mae:.4f}"
        )

    # ======================================================================
    # Scheme 3: TimeSeriesSplit CV
    # ======================================================================
    print("\n" + "=" * 70)
    print(f"{feature_type.upper()} | Scheme 3: TimeSeriesSplit CV (mean±std over folds)")
    print("=" * 70)

    tscv = TimeSeriesSplit(n_splits=5)

    for name, model in models.items():
        fold_r2s = []
        fold_mses = []
        fold_maes = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]

            r2, mse, mae = fit_and_evaluate(
                model, X_train_fold, y_train_fold, X_test_fold, y_test_fold
            )

            fold_r2s.append(r2)
            fold_mses.append(mse)
            fold_maes.append(mae)

            print(f"fold {fold}: r2={r2:.4f}, mse={mse:.4f}, mae={mae:.4f}")

        results[(3, name)] = {
            "r2_mean": pd.Series(fold_r2s).mean(),
            "r2_std": pd.Series(fold_r2s).std(),
            "mse_mean": pd.Series(fold_mses).mean(),
            "mse_std": pd.Series(fold_mses).std(),
            "mae_mean": pd.Series(fold_maes).mean(),
            "mae_std": pd.Series(fold_maes).std(),
        }

        print(
            f"{name}: "
            f"r2={results[(3, name)]['r2_mean']:.4f}±{results[(3, name)]['r2_std']:.4f}, "
            f"mse={results[(3, name)]['mse_mean']:.4f}±{results[(3, name)]['mse_std']:.4f}, "
            f"mae={results[(3, name)]['mae_mean']:.4f}±{results[(3, name)]['mae_std']:.4f}"
        )

    # ---------------------------
    # Convert results dict -> DataFrame
    # ---------------------------
    rows = []

    for (scheme, model), metrics in results.items():
        row = {
            "type": feature_type,
            "scheme": scheme,
            "model": model,
        }

        if scheme == 1:
            row["r2"] = metrics.get("r2")
            row["mse"] = metrics.get("mse")
            row["mae"] = metrics.get("mae")

        elif scheme == 2:
            row["val_r2"] = metrics.get("val_r2")
            row["val_mse"] = metrics.get("val_mse")
            row["val_mae"] = metrics.get("val_mae")
            row["test_r2"] = metrics.get("test_r2")
            row["test_mse"] = metrics.get("test_mse")
            row["test_mae"] = metrics.get("test_mae")

        elif scheme == 3:
            row["r2_mean"] = metrics.get("r2_mean")
            row["r2_std"] = metrics.get("r2_std")
            row["mse_mean"] = metrics.get("mse_mean")
            row["mse_std"] = metrics.get("mse_std")
            row["mae_mean"] = metrics.get("mae_mean")
            row["mae_std"] = metrics.get("mae_std")

        rows.append(row)

    df_results = pd.DataFrame(rows)
    return df_results


# ---------------------------
# 7. Main
# ---------------------------
if __name__ == "__main__":
    df_base = run_all(feature_type="base", use_engineered_features=False)
    df_featured = run_all(feature_type="featured", use_engineered_features=True)

    df_all = pd.concat([df_base, df_featured], ignore_index=True)

    print("\n" + "=" * 70)
    print("FULL RESULTS DATAFRAME")
    print("=" * 70)
    print(df_all)

    # ---------------------------
    # Scheme 1 comparison
    # ---------------------------
    scheme1 = df_all[df_all["scheme"] == 1]
    pivot1 = scheme1.pivot_table(
        index="model",
        columns="type",
        values=["r2", "mse", "mae"]
    )

    print("\n" + "=" * 70)
    print("SCHEME 1 COMPARISON")
    print("=" * 70)
    print(pivot1)

    # ---------------------------
    # Scheme 2 comparison (validation)
    # ---------------------------
    scheme2 = df_all[df_all["scheme"] == 2]
    pivot2_val = scheme2.pivot_table(
        index="model",
        columns="type",
        values=["val_r2", "val_mse", "val_mae"]
    )

    print("\n" + "=" * 70)
    print("SCHEME 2 VALIDATION COMPARISON")
    print("=" * 70)
    print(pivot2_val)

    # ---------------------------
    # Scheme 2 comparison (test)
    # ---------------------------
    pivot2_test = scheme2.pivot_table(
        index="model",
        columns="type",
        values=["test_r2", "test_mse", "test_mae"]
    )

    print("\n" + "=" * 70)
    print("SCHEME 2 TEST COMPARISON")
    print("=" * 70)
    print(pivot2_test)

    # ---------------------------
    # Scheme 3 comparison
    # ---------------------------
    scheme3 = df_all[df_all["scheme"] == 3]
    pivot3 = scheme3.pivot_table(
        index="model",
        columns="type",
        values=["r2_mean", "mse_mean", "mae_mean"]
    )

    print("\n" + "=" * 70)
    print("SCHEME 3 COMPARISON")
    print("=" * 70)
    print(pivot3)

    # Save full long-form results
    df_all.to_csv("day19results.csv", index=False)

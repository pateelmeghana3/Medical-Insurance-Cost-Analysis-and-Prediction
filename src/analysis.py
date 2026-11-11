"""
Medical Insurance Cost Analysis & Prediction
--------------------------------------------
Pipeline:  Load -> Clean -> EDA -> Hypothesis Tests -> Regression -> Evaluate -> Save

Run from the project root:
    python src/analysis.py
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed; saves figures to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "insurance.csv"
FIG_DIR = ROOT / "outputs" / "figures"
OUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
for d in (FIG_DIR, OUT_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
SEED = 42
sns.set_theme(style="whitegrid")


# ================================================================ 1. LOAD
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download insurance.csv from Kaggle "
            "('Medical Cost Personal Datasets') and place it in the data/ folder."
        )
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# ================================================================ 2. CLEAN
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    report = {
        "rows_before": len(df),
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    for col in ("sex", "smoker", "region"):
        df[col] = df[col].str.strip().str.lower()
    report["rows_after"] = len(df)

    # outliers in BMI by the IQR rule (flagged, not removed)
    q1, q3 = df["bmi"].quantile([0.25, 0.75])
    iqr = q3 - q1
    report["bmi_outliers_flagged"] = int(
        ((df["bmi"] < q1 - 1.5 * iqr) | (df["bmi"] > q3 + 1.5 * iqr)).sum()
    )
    print("Cleaning report:", report)
    pd.Series(report).to_csv(OUT_DIR / "cleaning_report.csv", header=["value"])
    return df


# ================================================================ 3. EDA
def run_eda(df: pd.DataFrame) -> None:
    df.describe().T.to_csv(OUT_DIR / "descriptive_statistics.csv")
    print("\nSkewness of charges:", round(df["charges"].skew(), 2))

    # distribution of target (raw vs log)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["charges"], kde=True, ax=ax[0], color="steelblue")
    ax[0].set_title("Distribution of Charges (right-skewed)")
    sns.histplot(np.log(df["charges"]), kde=True, ax=ax[1], color="seagreen")
    ax[1].set_title("Distribution of log(Charges)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_charges_distribution.png", dpi=150)
    plt.close(fig)

    # smoker vs charges
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="smoker", y="charges", palette="Set2", hue="smoker", legend=False)
    plt.title("Charges: Smokers vs Non-smokers")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_smoker_vs_charges.png", dpi=150)
    plt.close()

    # age / bmi vs charges coloured by smoker
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.scatterplot(data=df, x="age", y="charges", hue="smoker", alpha=0.6, ax=ax[0])
    ax[0].set_title("Age vs Charges")
    sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker", alpha=0.6, ax=ax[1])
    ax[1].set_title("BMI vs Charges")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_age_bmi_vs_charges.png", dpi=150)
    plt.close(fig)

    # correlation heatmap (numeric + encoded smoker)
    corr_df = df.copy()
    corr_df["smoker_code"] = (corr_df["smoker"] == "yes").astype(int)
    corr = corr_df[["age", "bmi", "children", "smoker_code", "charges"]].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_correlation_heatmap.png", dpi=150)
    plt.close()


# ================================================================ 4. HYPOTHESIS TESTS
def cohens_d(a, b) -> float:
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled


def run_hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # H1: smokers pay different charges (Welch's two-sample t-test)
    smk = df.loc[df["smoker"] == "yes", "charges"]
    non = df.loc[df["smoker"] == "no", "charges"]
    t, p = stats.ttest_ind(smk, non, equal_var=False)
    rows.append({
        "Test": "Welch t-test",
        "H0": "Mean charges of smokers = non-smokers",
        "Statistic": round(t, 3), "p_value": p,
        "Effect size": f"Cohen's d = {cohens_d(smk, non):.2f}",
    })

    # H2: charges differ between sexes
    m = df.loc[df["sex"] == "male", "charges"]
    f = df.loc[df["sex"] == "female", "charges"]
    t, p = stats.ttest_ind(m, f, equal_var=False)
    rows.append({
        "Test": "Welch t-test",
        "H0": "Mean charges of males = females",
        "Statistic": round(t, 3), "p_value": p,
        "Effect size": f"Cohen's d = {cohens_d(m, f):.2f}",
    })

    # H3: charges differ across regions (one-way ANOVA)
    groups = [g["charges"].values for _, g in df.groupby("region")]
    F, p = stats.f_oneway(*groups)
    ss_between = sum(len(g) * (g.mean() - df["charges"].mean()) ** 2 for g in groups)
    ss_total = ((df["charges"] - df["charges"].mean()) ** 2).sum()
    rows.append({
        "Test": "One-way ANOVA",
        "H0": "Mean charges equal across 4 regions",
        "Statistic": round(F, 3), "p_value": p,
        "Effect size": f"eta^2 = {ss_between / ss_total:.3f}",
    })

    # H4: smoking status is independent of sex (chi-square)
    table = pd.crosstab(df["sex"], df["smoker"])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    rows.append({
        "Test": "Chi-square independence",
        "H0": "Smoker status independent of sex",
        "Statistic": round(chi2, 3), "p_value": p,
        "Effect size": f"Cramer's V = {cramers_v:.3f}",
    })

    # H5: correlation between BMI and charges (Pearson)
    r, p = stats.pearsonr(df["bmi"], df["charges"])
    rows.append({
        "Test": "Pearson correlation",
        "H0": "No linear correlation between BMI and charges",
        "Statistic": round(r, 3), "p_value": p,
        "Effect size": f"r = {r:.2f}",
    })

    # normality of charges (justifies log transform / non-parametric check)
    sample = df["charges"].sample(min(len(df), 5000), random_state=SEED)
    w, p = stats.shapiro(sample)
    rows.append({
        "Test": "Shapiro-Wilk normality",
        "H0": "Charges are normally distributed",
        "Statistic": round(w, 3), "p_value": p,
        "Effect size": "-",
    })

    # robustness: Mann-Whitney U for smokers vs non-smokers
    u, p = stats.mannwhitneyu(smk, non, alternative="two-sided")
    rows.append({
        "Test": "Mann-Whitney U (robustness)",
        "H0": "Charge distributions of smokers = non-smokers",
        "Statistic": round(u, 1), "p_value": p,
        "Effect size": "-",
    })

    res = pd.DataFrame(rows)
    res["Decision (alpha=0.05)"] = np.where(res["p_value"] < ALPHA, "Reject H0", "Fail to reject H0")
    res["p_value"] = res["p_value"].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
    res.to_csv(OUT_DIR / "hypothesis_test_results.csv", index=False)
    print("\nHYPOTHESIS TESTS\n", res.to_string(index=False))

    # 95% CI for mean difference (smoker - non-smoker)
    diff = smk.mean() - non.mean()
    se = np.sqrt(smk.var(ddof=1) / len(smk) + non.var(ddof=1) / len(non))
    ci = (diff - 1.96 * se, diff + 1.96 * se)
    print(f"\nMean charge difference (smoker - non-smoker): {diff:,.0f}  95% CI [{ci[0]:,.0f}, {ci[1]:,.0f}]")
    return res


# ================================================================ 5. MODELLING
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: simple, explainable additions."""
    out = df.copy()
    out["bmi_obese"] = (out["bmi"] >= 30).astype(int)
    out["smoker_x_bmi"] = (out["smoker"] == "yes").astype(int) * out["bmi"]
    out["age_sq"] = out["age"] ** 2
    return out


def make_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ])


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def run_modelling(df: pd.DataFrame) -> None:
    data = build_features(df)
    y = data["charges"]
    X = data.drop(columns="charges")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=X["smoker"]
    )

    # --- model A: baseline linear regression (no engineered features)
    base_num = ["age", "bmi", "children"]
    base_cat = ["sex", "smoker", "region"]
    baseline = Pipeline([
        ("prep", make_preprocessor(base_num, base_cat)),
        ("model", LinearRegression()),
    ])
    baseline.fit(X_train[base_num + base_cat], y_train)

    # --- model B: improved linear regression (engineered features)
    imp_num = ["age", "bmi", "children", "age_sq", "smoker_x_bmi", "bmi_obese"]
    improved = Pipeline([
        ("prep", make_preprocessor(imp_num, base_cat)),
        ("model", LinearRegression()),
    ])
    improved.fit(X_train[imp_num + base_cat], y_train)

    # --- model C: random forest (non-linear benchmark)
    rf = Pipeline([
        ("prep", make_preprocessor(imp_num, base_cat)),
        ("model", RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=SEED, n_jobs=-1)),
    ])
    rf.fit(X_train[imp_num + base_cat], y_train)

    models = {
        "Linear Regression (baseline)": (baseline, base_num + base_cat),
        "Linear Regression (engineered features)": (improved, imp_num + base_cat),
        "Random Forest": (rf, imp_num + base_cat),
    }

    rows = []
    preds = {}
    for name, (mdl, cols) in models.items():
        p = mdl.predict(X_test[cols])
        preds[name] = p
        cv = cross_val_score(mdl, X_train[cols], y_train, cv=5, scoring="r2")
        r2 = r2_score(y_test, p)
        n, k = len(y_test), len(cols)
        rows.append({
            "Model": name,
            "R2 (test)": round(r2, 4),
            "Adj R2 (test)": round(1 - (1 - r2) * (n - 1) / (n - k - 1), 4),
            "RMSE": round(rmse(y_test, p), 1),
            "MAE": round(mean_absolute_error(y_test, p), 1),
            "CV R2 (5-fold mean)": round(cv.mean(), 4),
        })
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print("\nMODEL COMPARISON\n", results.to_string(index=False))

    best_name = results.sort_values("RMSE").iloc[0]["Model"]
    best_model, best_cols = models[best_name]
    best_pred = preds[best_name]
    print(f"\nBest model by RMSE: {best_name}")

    # --- diagnostics: actual vs predicted + residuals
    resid = y_test - best_pred
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].scatter(y_test, best_pred, alpha=0.6, color="steelblue")
    lim = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
    ax[0].plot(lim, lim, "r--")
    ax[0].set(title="Actual vs Predicted", xlabel="Actual", ylabel="Predicted")
    ax[1].scatter(best_pred, resid, alpha=0.6, color="darkorange")
    ax[1].axhline(0, color="r", ls="--")
    ax[1].set(title="Residuals vs Predicted", xlabel="Predicted", ylabel="Residual")
    stats.probplot(resid, dist="norm", plot=ax[2])
    ax[2].set_title("Residual Q-Q Plot")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_model_diagnostics.png", dpi=150)
    plt.close(fig)

    # --- model comparison bar chart
    plt.figure(figsize=(8, 4))
    sns.barplot(data=results, x="R2 (test)", y="Model", palette="viridis", hue="Model", legend=False)
    plt.xlim(0, 1)
    plt.title("Model Comparison (Test R²)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_model_comparison.png", dpi=150)
    plt.close()

    # --- coefficients of the improved linear model (interpretation)
    lin = improved.named_steps["model"]
    names = improved.named_steps["prep"].get_feature_names_out()
    coefs = pd.DataFrame({"feature": names, "coefficient": lin.coef_}).sort_values(
        "coefficient", key=abs, ascending=False)
    coefs.to_csv(OUT_DIR / "linear_model_coefficients.csv", index=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=coefs, x="coefficient", y="feature", palette="coolwarm", hue="feature", legend=False)
    plt.title("Linear Model Coefficients (standardised features)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_coefficients.png", dpi=150)
    plt.close()

    # --- feature importance from the forest
    fi = pd.DataFrame({
        "feature": rf.named_steps["prep"].get_feature_names_out(),
        "importance": rf.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False).head(10)
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=fi, x="importance", y="feature", palette="mako", hue="feature", legend=False)
    plt.title("Random Forest Feature Importance (Top 10)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_feature_importance.png", dpi=150)
    plt.close()

    # --- residual normality check
    w, p = stats.shapiro(resid)
    print(f"Residual Shapiro-Wilk: W={w:.3f}, p={p:.4g}")

    # --- save best model
    joblib.dump({"model": best_model, "columns": best_cols}, MODEL_DIR / "best_model.joblib")
    print(f"Saved model -> {MODEL_DIR / 'best_model.joblib'}")


# ================================================================ MAIN
def main() -> None:
    df = load_data(DATA_PATH)
    df = clean_data(df)
    run_eda(df)
    run_hypothesis_tests(df)
    run_modelling(df)
    print("\nDone. See the outputs/ folder for tables and figures.")


if __name__ == "__main__":
    main()

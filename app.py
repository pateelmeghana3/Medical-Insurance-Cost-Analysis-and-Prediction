"""
Streamlit dashboard for the Medical Insurance Cost project.

Run (from the project root, after `python src/analysis.py` has been run once):
    streamlit run app.py
"""

import math
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from analysis import (  # noqa: E402
    DATA_PATH,
    FIG_DIR,
    MODEL_DIR,
    OUT_DIR,
    build_features,
    clean_data,
    cohens_d,
    load_data,
)

st.set_page_config(page_title="Insurance Cost Dashboard", page_icon="📊", layout="wide")
sns.set_theme(style="whitegrid")


# ------------------------------------------------------------------ helpers
@st.cache_data
def get_data() -> pd.DataFrame:
    return clean_data(load_data(DATA_PATH))


@st.cache_resource
def get_model():
    path = MODEL_DIR / "best_model.joblib"
    return joblib.load(path) if path.exists() else None


def show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def money(x: float) -> str:
    return f"${x:,.0f}"


try:
    df = get_data()
except FileNotFoundError:
    st.error("`data/insurance.csv` not found. Download it from Kaggle and place it in the `data/` folder.")
    st.stop()

# ------------------------------------------------------------------ sidebar filters
st.sidebar.title("Filters")
st.sidebar.caption("These filters apply to the Overview and Explore tabs.")
smoker_f = st.sidebar.multiselect("Smoker", ["yes", "no"], default=["yes", "no"])
sex_f = st.sidebar.multiselect("Sex", ["male", "female"], default=["male", "female"])
regions = sorted(df["region"].unique())
region_f = st.sidebar.multiselect("Region", regions, default=regions)
age_min, age_max = int(df["age"].min()), int(df["age"].max())
age_f = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
bmi_min, bmi_max = float(math.floor(df["bmi"].min())), float(math.ceil(df["bmi"].max()))
bmi_f = st.sidebar.slider("BMI range", bmi_min, bmi_max, (bmi_min, bmi_max))

mask = (
    df["smoker"].isin(smoker_f)
    & df["sex"].isin(sex_f)
    & df["region"].isin(region_f)
    & df["age"].between(*age_f)
    & df["bmi"].between(*bmi_f)
)
fdf = df[mask]

# ------------------------------------------------------------------ header
st.title("📊 Medical Insurance Cost: Analysis and Prediction")
st.caption("Statistics + regression project | Dataset: Kaggle Medical Cost Personal Datasets")

tab_over, tab_eda, tab_test, tab_model, tab_pred = st.tabs(
    ["Overview", "Explore", "Hypothesis Tests", "Model Results", "Predict Charges"]
)

# ================================================================== OVERVIEW
with tab_over:
    if fdf.empty:
        st.warning("No records match the current filters. Widen the filters in the sidebar.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records", f"{len(fdf):,}")
        c2.metric("Average charges", money(fdf["charges"].mean()))
        c3.metric("Smokers", f"{(fdf['smoker'] == 'yes').mean() * 100:.1f}%")
        c4.metric("Average BMI", f"{fdf['bmi'].mean():.1f}")

        s = fdf.loc[fdf["smoker"] == "yes", "charges"]
        n = fdf.loc[fdf["smoker"] == "no", "charges"]
        if len(s) and len(n):
            st.info(
                f"In the selected data, smokers pay on average **{money(s.mean())}** versus "
                f"**{money(n.mean())}** for non-smokers, about **{s.mean() / n.mean():.1f}x** more."
            )

        left, right = st.columns(2)
        with left:
            st.subheader("Charges distribution")
            fig, ax = plt.subplots(figsize=(6, 3.8))
            sns.histplot(fdf["charges"], kde=True, ax=ax, color="steelblue")
            ax.set_xlabel("Charges ($)")
            show(fig)
        with right:
            st.subheader("Average charges by group")
            by = st.selectbox("Group by", ["smoker", "region", "sex", "children"], key="ov_group")
            grp = fdf.groupby(by)["charges"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(6, 3.8))
            sns.barplot(data=grp, x=by, y="charges", hue=by, palette="viridis", legend=False, ax=ax)
            ax.set_ylabel("Average charges ($)")
            show(fig)

        with st.expander("View data"):
            st.dataframe(fdf)
            st.download_button("Download filtered data (CSV)", fdf.to_csv(index=False), "filtered_insurance.csv")

# ================================================================== EXPLORE
with tab_eda:
    if fdf.empty:
        st.warning("No records match the current filters.")
    else:
        chart = st.selectbox(
            "Choose a chart",
            [
                "Charges by category (box plot)",
                "Scatter: numeric variable vs charges",
                "Correlation heatmap",
                "Distribution of a numeric variable",
            ],
        )
        if chart == "Charges by category (box plot)":
            cat = st.selectbox("Category", ["smoker", "sex", "region", "children"], key="eda_cat")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.boxplot(data=fdf, x=cat, y="charges", hue=cat, palette="Set2", legend=False, ax=ax)
            ax.set_ylabel("Charges ($)")
            show(fig)
        elif chart == "Scatter: numeric variable vs charges":
            x = st.selectbox("X axis", ["age", "bmi"], key="eda_x")
            hue = st.selectbox("Colour by", ["smoker", "sex", "region"], key="eda_hue")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.scatterplot(data=fdf, x=x, y="charges", hue=hue, alpha=0.65, ax=ax)
            ax.set_ylabel("Charges ($)")
            show(fig)
            st.caption("Tip: colour by smoker. Smokers form a separate, higher-cost cluster.")
        elif chart == "Correlation heatmap":
            tmp = fdf.copy()
            tmp["smoker_code"] = (tmp["smoker"] == "yes").astype(int)
            corr = tmp[["age", "bmi", "children", "smoker_code", "charges"]].corr()
            fig, ax = plt.subplots(figsize=(6, 4.8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
            show(fig)
        else:
            col = st.selectbox("Variable", ["charges", "age", "bmi"], key="eda_dist")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.histplot(fdf[col], kde=True, ax=ax, color="seagreen")
            show(fig)
        st.subheader("Descriptive statistics")
        st.dataframe(fdf[["age", "bmi", "children", "charges"]].describe().T)

# ================================================================== HYPOTHESIS TESTS
with tab_test:
    st.write(
        "Pick a question and the matching statistical test runs on the **full dataset** "
        "(not the sidebar filters), so the result stays valid."
    )
    alpha = st.select_slider("Significance level (alpha)", options=[0.01, 0.05, 0.10], value=0.05)
    question = st.selectbox(
        "Question",
        [
            "Do smokers and non-smokers pay different charges? (Welch t-test)",
            "Do males and females pay different charges? (Welch t-test)",
            "Do charges differ across regions? (One-way ANOVA)",
            "Is smoking related to sex? (Chi-square test)",
            "Is BMI correlated with charges? (Pearson correlation)",
        ],
    )

    fig, ax = plt.subplots(figsize=(6.5, 4))
    if question.startswith("Do smokers"):
        a = df.loc[df["smoker"] == "yes", "charges"]
        b = df.loc[df["smoker"] == "no", "charges"]
        stat, p = stats.ttest_ind(a, b, equal_var=False)
        stat_name, effect = "t statistic", f"Cohen's d = {cohens_d(a, b):.2f}"
        h0 = "Mean charges of smokers = non-smokers"
        detail = f"Average charges: smokers {money(a.mean())} vs non-smokers {money(b.mean())}."
        sns.boxplot(data=df, x="smoker", y="charges", hue="smoker", palette="Set2", legend=False, ax=ax)
    elif question.startswith("Do males"):
        a = df.loc[df["sex"] == "male", "charges"]
        b = df.loc[df["sex"] == "female", "charges"]
        stat, p = stats.ttest_ind(a, b, equal_var=False)
        stat_name, effect = "t statistic", f"Cohen's d = {cohens_d(a, b):.2f}"
        h0 = "Mean charges of males = females"
        detail = f"Average charges: males {money(a.mean())} vs females {money(b.mean())}."
        sns.boxplot(data=df, x="sex", y="charges", hue="sex", palette="Set2", legend=False, ax=ax)
    elif "regions" in question:
        groups = [g["charges"].values for _, g in df.groupby("region")]
        stat, p = stats.f_oneway(*groups)
        ss_between = sum(len(g) * (g.mean() - df["charges"].mean()) ** 2 for g in groups)
        ss_total = ((df["charges"] - df["charges"].mean()) ** 2).sum()
        stat_name, effect = "F statistic", f"eta squared = {ss_between / ss_total:.3f}"
        h0 = "Mean charges are equal across all 4 regions"
        means = df.groupby("region")["charges"].mean()
        detail = "Average charges: " + ", ".join(f"{r} {money(v)}" for r, v in means.items()) + "."
        sns.boxplot(data=df, x="region", y="charges", hue="region", palette="Set2", legend=False, ax=ax)
    elif "related to sex" in question:
        table = pd.crosstab(df["sex"], df["smoker"])
        stat, p, _, _ = stats.chi2_contingency(table)
        v = math.sqrt(stat / (table.values.sum() * (min(table.shape) - 1)))
        stat_name, effect = "chi-square statistic", f"Cramer's V = {v:.3f}"
        h0 = "Smoking status is independent of sex"
        detail = "Counts shown in the chart (rows = sex, columns = smoker)."
        table.plot(kind="bar", ax=ax, color=["#8fd19e", "#f08a8a"], rot=0)
    else:
        r, p = stats.pearsonr(df["bmi"], df["charges"])
        stat, stat_name, effect = r, "correlation r", f"r = {r:.2f}"
        h0 = "There is no linear correlation between BMI and charges"
        detail = f"Correlation coefficient r = {r:.2f} (closer to 0 means weaker relationship)."
        sns.scatterplot(data=df, x="bmi", y="charges", alpha=0.5, ax=ax)

    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"**Null hypothesis (H0):** {h0}")
        m1, m2 = st.columns(2)
        m1.metric(stat_name, f"{stat:.3f}")
        m2.metric("p-value", f"{p:.2e}" if p < 0.001 else f"{p:.4f}")
        st.write(f"**Effect size:** {effect}")
        st.write(detail)
        if p < alpha:
            st.success(
                f"p < {alpha}, so we **reject H0**. The difference/relationship is statistically significant."
            )
        else:
            st.warning(
                f"p >= {alpha}, so we **fail to reject H0**. There is not enough evidence of an effect."
            )
        st.caption(
            "A small p-value says the effect is unlikely to be chance. The effect size says how big it is. "
            "Always read both."
        )
    with right:
        show(fig)

    results_path = OUT_DIR / "hypothesis_test_results.csv"
    if results_path.exists():
        with st.expander("Full hypothesis test table (from analysis.py)"):
            st.dataframe(pd.read_csv(results_path))

# ================================================================== MODEL RESULTS
with tab_model:
    comp_path = OUT_DIR / "model_comparison.csv"
    if not comp_path.exists():
        st.warning("Run `python src/analysis.py` first to generate model results.")
    else:
        comp = pd.read_csv(comp_path)
        best = comp.sort_values("RMSE").iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best model", best["Model"])
        c2.metric("Test R²", f"{best['R2 (test)']:.3f}")
        c3.metric("RMSE", money(best["RMSE"]))

        st.dataframe(comp)

        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
        sns.barplot(data=comp, x="R2 (test)", y="Model", hue="Model", palette="viridis", legend=False, ax=axes[0])
        axes[0].set_xlim(0, 1)
        axes[0].set_title("R² (higher is better)")
        sns.barplot(data=comp, x="RMSE", y="Model", hue="Model", palette="rocket", legend=False, ax=axes[1])
        axes[1].set_title("RMSE in $ (lower is better)")
        axes[1].set_ylabel("")
        axes[1].set_yticklabels([])
        fig.tight_layout()
        show(fig)

        st.subheader("Diagnostics and importance")
        images = [
            ("05_model_diagnostics.png", "Actual vs predicted, residuals and Q-Q plot"),
            ("07_coefficients.png", "Linear model coefficients"),
            ("08_feature_importance.png", "Random forest feature importance"),
        ]
        for fname, caption in images:
            img = FIG_DIR / fname
            if img.exists():
                st.image(str(img), caption=caption)

# ================================================================== PREDICT
with tab_pred:
    bundle = get_model()
    if bundle is None:
        st.warning("No saved model found. Run `python src/analysis.py` first.")
    else:
        st.write("Enter a profile to estimate annual medical insurance charges.")
        c1, c2, c3 = st.columns(3)
        age = c1.slider("Age", 18, 64, 35)
        bmi = c2.number_input("BMI", min_value=15.0, max_value=55.0, value=28.0, step=0.1)
        children = c3.slider("Children", 0, 5, 1)
        c4, c5, c6 = st.columns(3)
        sex = c4.selectbox("Sex", ["male", "female"], key="p_sex")
        smoker = c5.selectbox("Smoker", ["no", "yes"], key="p_smoker")
        region = c6.selectbox("Region", regions, key="p_region")

        def predict(smoker_value: str) -> float:
            row = pd.DataFrame([{
                "age": age, "sex": sex, "bmi": bmi, "children": children,
                "smoker": smoker_value, "region": region,
            }])
            row = build_features(row)
            return float(bundle["model"].predict(row[bundle["columns"]])[0])

        if st.button("Predict charges"):
            pred = predict(smoker)
            other = "yes" if smoker == "no" else "no"
            pred_other = predict(other)

            m1, m2 = st.columns(2)
            m1.metric("Estimated annual charges", money(pred))
            similar = df[(df["smoker"] == smoker) & (df["age"].between(age - 5, age + 5))]
            if len(similar):
                m2.metric(f"Average for similar people (age ±5, smoker = {smoker})", money(similar["charges"].mean()))

            st.subheader("What-if: effect of smoking status")
            what_if = pd.DataFrame({
                "Smoker": ["no", "yes"],
                "Estimated charges": [predict("no"), predict("yes")],
            })
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.barplot(data=what_if, x="Smoker", y="Estimated charges", hue="Smoker",
                        palette=["#8fd19e", "#f08a8a"], legend=False, ax=ax)
            show(fig)
            diff = pred_other - pred
            direction = "higher" if diff > 0 else "lower"
            st.write(
                f"For the same profile, switching smoker status to **{other}** would give a "
                f"prediction about **{money(abs(diff))} {direction}**."
            )
        st.caption(
            "This is an educational model trained on a small public US dataset. "
            "It is not a real insurance quote."
        )

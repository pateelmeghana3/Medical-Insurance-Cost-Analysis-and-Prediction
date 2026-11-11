# Project Report: Medical Insurance Cost Analysis and Prediction

**Student:** <name>  **Degree:** BSc Statistics  **College:** SV Degree College, Anantapur
**Tools:** Python (Pandas, NumPy, SciPy, Scikit-learn, Matplotlib, Seaborn)

---
## 1. Abstract
This project studies how personal factors (age, sex, BMI, number of children, smoking status and region)
affect medical insurance charges. Hypothesis tests establish which factors matter statistically, and
regression models predict charges. (Add one line with your best result: "The best model achieved R² of __.")

## 2. Introduction and Objectives
Insurance companies price policies based on risk. Understanding the drivers of cost helps with pricing and
with health awareness.
**Objectives**
1. Clean and explore the dataset.
2. Test hypotheses about factors affecting charges.
3. Build and compare regression models.
4. Identify the most influential variables.

## 3. Dataset Description
| Variable | Type | Description |
|---|---|---|
| age | numeric | Age of the primary beneficiary |
| sex | categorical | male / female |
| bmi | numeric | Body mass index |
| children | numeric | Number of dependants |
| smoker | categorical | yes / no |
| region | categorical | northeast, northwest, southeast, southwest |
| charges | numeric (target) | Annual medical costs billed by insurance |

## 4. Methodology
### 4.1 Data Cleaning
Checked missing values and duplicates, standardised text case, flagged BMI outliers using the IQR rule
(values beyond Q1 - 1.5 IQR and Q3 + 1.5 IQR). Outliers were kept because they are valid observations.

### 4.2 Exploratory Data Analysis
Charges are right-skewed (skewness reported in the output), which is why a Welch test and a Mann-Whitney
robustness check are used. Scatter plots show smokers form a separate, higher-cost cluster.

### 4.3 Hypothesis Testing
Significance level alpha = 0.05.
| # | Test | H0 | H1 |
|---|---|---|---|
| 1 | Welch t-test | mean charges smoker = non-smoker | means differ |
| 2 | Welch t-test | mean charges male = female | means differ |
| 3 | One-way ANOVA | mean charges equal across regions | at least one region differs |
| 4 | Chi-square | smoker and sex independent | associated |
| 5 | Pearson | no correlation BMI and charges | correlated |
| 6 | Shapiro-Wilk | charges normal | not normal |
| 7 | Mann-Whitney U | smoker/non-smoker distributions equal | differ |

Effect sizes are reported with every test (Cohen's d, eta squared, Cramer's V, r), because a small
p-value alone does not show practical importance.

### 4.4 Feature Engineering
`bmi_obese` (BMI >= 30), `smoker x BMI` interaction and `age^2`. The interaction captures the fact that
BMI matters much more for smokers.

### 4.5 Modelling
- 80/20 train-test split, stratified by smoker, random seed 42
- Numeric features standardised, categorical features one-hot encoded (all inside a Scikit-learn Pipeline,
  so there is no data leakage)
- Models: Linear Regression (baseline), Linear Regression (engineered features), Random Forest
- Metrics: R², Adjusted R², RMSE, MAE, 5-fold cross-validated R²

### 4.6 Model Diagnostics
Actual vs predicted plot, residuals vs predicted (checks constant variance), Q-Q plot and Shapiro-Wilk test
on residuals (checks normality).

## 5. Results
(Paste your tables from `outputs/` and add figures from `outputs/figures/`.)
- 5.1 Descriptive statistics
- 5.2 Hypothesis test table and interpretation of each decision
- 5.3 Model comparison table
- 5.4 Best-model diagnostics
- 5.5 Most important features

## 6. Discussion
Explain in plain language: which factor mattered most, whether region/sex mattered, why the engineered
model did better than the baseline, and where predictions are weakest.

## 7. Limitations
Small dataset, US data, no medical history variables, charges are skewed.

## 8. Conclusion and Future Work
Summarise findings in 3 lines. Future work: gradient boosting, log-transformed target, Streamlit app,
Power BI dashboard.

## 9. References
- Dataset: Kaggle, Medical Cost Personal Datasets (Miri Choi)
- Scikit-learn and SciPy documentation

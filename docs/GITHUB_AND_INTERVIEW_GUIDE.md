# Step-by-Step: Build, Push to GitHub, and Prepare for Interviews

## Part A: Setup (Windows)
1. Install **Python 3.10+** from python.org (tick "Add to PATH").
2. Install **Git** from git-scm.com and **VS Code** from code.visualstudio.com.
3. Create a free **GitHub** account (use a professional username).
4. Download `insurance.csv` from Kaggle ("Medical Cost Personal Datasets") and put it in `data/`.

## Part B: Run the project
```bash
cd insurance-cost-analysis
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/analysis.py
python src/predict.py --age 35 --sex male --bmi 28.5 --children 2 --smoker no --region southeast
```
Open `outputs/` and check the CSV tables and `outputs/figures/` images.
Read the terminal output and understand every number before moving on.

## Part B2: Run the dashboard
```bash
streamlit run app.py
```
Your browser opens the dashboard. Press `Ctrl + C` in the terminal to stop it.
Take a screenshot (Windows key + Shift + S) and save it as `docs/dashboard.png`, then commit it so the README shows it.

## Part C: Fill in your results
1. Open `outputs/hypothesis_test_results.csv` and `outputs/model_comparison.csv`.
2. Copy the numbers into the Results section of `README.md` and `docs/PROJECT_REPORT.md`.
3. Write the 3 to 4 "Key findings" lines in your own words.
4. Replace `<Your name>` and the profile links.

## Part D: Push to GitHub
1. On github.com click **New repository**. Name: `insurance-cost-analysis`. Public. Do NOT tick "Add README".
2. In the project folder run:
```bash
git init
git add .
git commit -m "Initial commit: insurance cost analysis project"
git branch -M main
git remote add origin https://github.com/<your-username>/insurance-cost-analysis.git
git push -u origin main
```
3. On the repo page: add a short **About** description and topics
   (`python`, `statistics`, `regression`, `hypothesis-testing`, `data-analysis`, `scikit-learn`).
4. Check the README shows the diagram and results table properly.
5. Pin the repo on your GitHub profile.

Tip: commit in small steps (for example "add EDA", "add hypothesis tests", "add models") so the history looks natural.
Also commit your own `outputs/figures/` so recruiters can see the charts without running anything.

## Part E: Resume entry (copy and edit with YOUR numbers)
**Medical Insurance Cost Analysis and Prediction** | Python, Pandas, SciPy, Scikit-learn, Matplotlib
- Cleaned and analysed 1,338 insurance records, handling duplicates, missing values and outliers.
- Performed hypothesis tests (t-test, ANOVA, chi-square, correlation) and found smoking status as the most significant cost driver (p < 0.001).
- Built and compared linear regression and random forest models, achieving R² of __ and RMSE of __ on test data.
- Published the full code, report and visualisations on GitHub: github.com/<username>/insurance-cost-analysis

## Part F: Interview questions you must be able to answer
1. **Why a Welch t-test instead of a normal t-test?** It does not assume equal variances, and smoker/non-smoker variances clearly differ.
2. **Charges are not normal. Is a t-test still valid?** With large samples the CLT makes the mean approximately normal; I also ran Mann-Whitney U as a non-parametric check.
3. **What does p < 0.05 mean?** If H0 were true, getting a result this extreme would happen less than 5% of the time. It is not the probability that H0 is true.
4. **Why report effect size?** A tiny p-value can come from a large sample even when the difference is trivial; effect size shows practical importance.
5. **What is R²? Adjusted R²?** R² is the share of variance explained; adjusted R² penalises extra predictors.
6. **Why did you use a Pipeline?** The scaler and encoder are fitted only on training data, which prevents data leakage.
7. **What is the smoker x BMI interaction?** The effect of BMI on cost depends on smoking status; smokers' costs rise much faster with BMI.
8. **How did you check regression assumptions?** Residuals vs predicted (constant variance), Q-Q plot and Shapiro-Wilk (normality).
9. **Why stratify the split by smoker?** So train and test sets have the same smoker proportion.
10. **What would you improve?** Log-transform the target, try gradient boosting, tune hyperparameters, add an app.
11. **RMSE vs MAE?** RMSE penalises large errors more; MAE is the average absolute error in rupee/dollar terms.
12. **What is multicollinearity?** Predictors correlated with each other; it makes coefficients unstable. Here the predictors are only weakly correlated.

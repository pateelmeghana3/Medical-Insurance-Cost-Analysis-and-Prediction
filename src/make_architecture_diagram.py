"""Generates docs/architecture.png (the pipeline / architecture diagram)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "architecture.png"

fig, ax = plt.subplots(figsize=(14, 6.2))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6.2)
ax.axis("off")

def box(x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                                fc=color, ec="#333333", lw=1.4))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center", fontsize=10.5, fontweight="bold")
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.7 - i * 0.3, t, ha="center", va="center", fontsize=8.5)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#333333"))

ax.text(7, 5.9, "Medical Insurance Cost Analysis - Pipeline Architecture", ha="center",
        fontsize=14, fontweight="bold")

# top row
box(0.3, 3.4, 2.4, 1.9, "1. Data Source", ["Kaggle insurance.csv", "1,338 records", "7 columns"], "#dbeafe")
box(3.2, 3.4, 2.4, 1.9, "2. Cleaning", ["Duplicates / nulls", "Text standardising", "IQR outlier flags"], "#dcfce7")
box(6.1, 3.4, 2.4, 1.9, "3. EDA", ["Descriptive stats", "Distributions", "Correlation heatmap"], "#fef9c3")
box(9.0, 3.4, 2.4, 1.9, "4. Hypothesis Tests", ["t-test, ANOVA", "Chi-square, Pearson", "Shapiro, Mann-Whitney"], "#fde2e2")
box(11.5, 3.4, 2.2, 1.9, "5. Features", ["BMI obese flag", "smoker x BMI", "age squared"], "#ede9fe")

for x in (2.7, 5.6, 8.5, 11.4):
    arrow(x + 0.02, 4.35, x + 0.5, 4.35)

# bottom row
box(0.3, 0.5, 3.0, 1.9, "6. Train / Test Split", ["80 / 20, seed = 42", "Stratified by smoker", "5-fold cross-validation"], "#e0f2fe")
box(4.0, 0.5, 3.4, 1.9, "7. Models (sklearn Pipeline)", ["Linear Regression (baseline)", "Linear Regression (engineered)", "Random Forest"], "#ffedd5")
box(8.1, 0.5, 2.7, 1.9, "8. Evaluation", ["R2, Adj R2, RMSE, MAE", "Residual diagnostics", "Best model selected"], "#fce7f3")
box(11.3, 0.5, 2.4, 1.9, "9. Outputs", ["best_model.joblib", "Figures + CSV tables", "predict.py (CLI)"], "#d1fae5")

arrow(12.6, 3.4, 1.8, 2.4)   # features -> split
arrow(3.3, 1.45, 4.0, 1.45)
arrow(7.4, 1.45, 8.1, 1.45)
arrow(10.8, 1.45, 11.3, 1.45)

fig.tight_layout()
fig.savefig(OUT, dpi=170)
print("Saved", OUT)

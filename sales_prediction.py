# ============================================================
#   SALES PREDICTION — CodeAlpha Internship Task 4
#   Author: Your Name
#   Description: Predict Sales based on Advertising spend
#                across TV, Radio and Newspaper platforms
# ============================================================

# ─────────────────────────────────────────
# STEP 1: Import Libraries
# ─────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("   SALES PREDICTION — CodeAlpha Task 4")
print("=" * 60)


# ─────────────────────────────────────────
# STEP 2: Load & Clean Dataset
# ─────────────────────────────────────────
df = pd.read_csv('Advertising.csv')

# Drop unnamed index column
df = df.drop('Unnamed: 0', axis=1)

print("\n📂 Dataset Loaded Successfully!")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")


# ─────────────────────────────────────────
# STEP 3: Explore the Data
# ─────────────────────────────────────────
print("\n─── First 5 Rows ───────────────────────────────────────")
print(df.head())

print("\n─── Missing Values ──────────────────────────────────────")
print(df.isnull().sum())

print("\n─── Statistical Summary ─────────────────────────────────")
print(df.describe())


# ─────────────────────────────────────────
# STEP 4: Data Visualization
# ─────────────────────────────────────────
print("\n📊 Generating Visualizations...")

# Plot 1 — Distribution of all columns
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Distribution of Advertising Budget & Sales", fontsize=14, fontweight='bold')
colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0']

for ax, col, color in zip(axes, df.columns, colors):
    ax.hist(df[col], bins=20, color=color, edgecolor='white', alpha=0.85)
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("sales_distributions.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: sales_distributions.png")

# Plot 2 — Advertising spend vs Sales (scatter plots)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Advertising Spend vs Sales", fontsize=14, fontweight='bold')
platforms = ['TV', 'Radio', 'Newspaper']
colors    = ['#2196F3', '#4CAF50', '#FF5722']

for ax, platform, color in zip(axes, platforms, colors):
    ax.scatter(df[platform], df['Sales'], color=color, alpha=0.6, edgecolors='white', linewidth=0.5)
    # Add trend line
    z = np.polyfit(df[platform], df['Sales'], 1)
    p = np.poly1d(z)
    ax.plot(sorted(df[platform]), p(sorted(df[platform])),
            color='red', linewidth=2, linestyle='--', label='Trend')
    ax.set_title(f"{platform} Spend vs Sales", fontweight='bold')
    ax.set_xlabel(f"{platform} Budget ($000s)")
    ax.set_ylabel("Sales ($000s)")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("sales_vs_advertising.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: sales_vs_advertising.png")

# Plot 3 — Correlation Heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="YlGnBu",
            linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("sales_correlation.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: sales_correlation.png")

# Plot 4 — Advertising Budget Breakdown (Pie Chart)
plt.figure(figsize=(7, 7))
budget_totals = df[['TV', 'Radio', 'Newspaper']].sum()
colors = ['#2196F3', '#4CAF50', '#FF5722']
explode = (0.05, 0.05, 0.05)
plt.pie(budget_totals, labels=budget_totals.index, autopct='%1.1f%%',
        colors=colors, explode=explode, startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
plt.title("Total Advertising Budget Distribution\nby Platform", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("sales_budget_breakdown.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: sales_budget_breakdown.png")


# ─────────────────────────────────────────
# STEP 5: Prepare Data for ML
# ─────────────────────────────────────────
print("\n⚙️  Preparing Data for Machine Learning...")

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing  samples : {X_test.shape[0]}")


# ─────────────────────────────────────────
# STEP 6: Train Multiple Models
# ─────────────────────────────────────────
print("\n🤖 Training Models...")

models = {
    "Linear Regression"       : LinearRegression(),
    "Random Forest Regressor" : RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting"       : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    results[name] = {"model": model, "predictions": y_pred, "MAE": mae, "RMSE": rmse, "R2": r2}
    print(f"   ✅ {name:<30} R²: {r2:.4f}  |  MAE: {mae:.2f}  |  RMSE: {rmse:.2f}")


# ─────────────────────────────────────────
# STEP 7: Best Model — Detailed Evaluation
# ─────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]["R2"])
best_preds = results[best_name]["predictions"]
best_model = results[best_name]["model"]

print(f"\n🏆 Best Model : {best_name}")
print(f"   R² Score   : {results[best_name]['R2']:.4f}  ({results[best_name]['R2']*100:.2f}% variance explained)")
print(f"   MAE        : {results[best_name]['MAE']:.2f} ($000s)")
print(f"   RMSE       : {results[best_name]['RMSE']:.2f} ($000s)")

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_preds, alpha=0.7, color='#2196F3', edgecolors='white', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linewidth=2, linestyle='--', label='Perfect Prediction')
plt.title(f"Actual vs Predicted Sales\n({best_name})", fontsize=13, fontweight='bold')
plt.xlabel("Actual Sales ($000s)")
plt.ylabel("Predicted Sales ($000s)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("sales_actual_vs_predicted.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: sales_actual_vs_predicted.png")


# ─────────────────────────────────────────
# STEP 8: Feature Importance
# ─────────────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(8, 4))
    bars = plt.barh(importances.index, importances.values,
                    color=['#FF5722', '#4CAF50', '#2196F3'], edgecolor='white')
    plt.title(f"Feature Importance — Which Platform Drives Sales Most?\n({best_name})",
              fontsize=12, fontweight='bold')
    plt.xlabel("Importance Score")
    for bar, val in zip(bars, importances.values):
        plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig("sales_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("   ✅ Saved: sales_feature_importance.png")


# ─────────────────────────────────────────
# STEP 9: Model Comparison Chart
# ─────────────────────────────────────────
model_names = list(results.keys())
r2_scores   = [results[m]["R2"] * 100 for m in model_names]
bar_colors  = ['#4CAF50' if m == best_name else '#90CAF9' for m in model_names]

plt.figure(figsize=(9, 5))
bars = plt.bar(model_names, r2_scores, color=bar_colors, edgecolor='white', width=0.5)
plt.ylim(0, 110)
plt.title("Model R² Score Comparison", fontsize=14, fontweight='bold')
plt.ylabel("R² Score (%)")
plt.xlabel("Model")
for bar, val in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f"{val:.2f}%", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("sales_model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: sales_model_comparison.png")


# ─────────────────────────────────────────
# STEP 10: Actionable Business Insights
# ─────────────────────────────────────────
print("\n─── 📋 Actionable Business Insights ───────────────────")
tv_corr   = df['TV'].corr(df['Sales'])
radio_corr = df['Radio'].corr(df['Sales'])
news_corr  = df['Newspaper'].corr(df['Sales'])
print(f"   • TV correlation with Sales        : {tv_corr:.4f}")
print(f"   • Radio correlation with Sales     : {radio_corr:.4f}")
print(f"   • Newspaper correlation with Sales : {news_corr:.4f}")

best_platform = max({'TV': tv_corr, 'Radio': radio_corr, 'Newspaper': news_corr},
                     key=lambda k: {'TV': tv_corr, 'Radio': radio_corr, 'Newspaper': news_corr}[k])
print(f"\n   💡 Most Impactful Platform → {best_platform}")
print(f"   💡 Invest more in {best_platform} advertising to maximize sales!")


# ─────────────────────────────────────────
# STEP 11: Predict Sales for New Budget
# ─────────────────────────────────────────
print("\n💰 Predicting Sales for a New Advertising Budget...")

new_budget = pd.DataFrame([[200.0, 40.0, 50.0]], columns=['TV', 'Radio', 'Newspaper'])
predicted_sales = best_model.predict(new_budget)[0]
print(f"   TV Budget        : $200,000")
print(f"   Radio Budget     : $40,000")
print(f"   Newspaper Budget : $50,000")
print(f"   Predicted Sales  → ${predicted_sales:.2f} (in $000s) = ${predicted_sales*1000:,.0f}")


# ─────────────────────────────────────────
# DONE
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("   ✅ ALL TASKS COMPLETED SUCCESSFULLY!")
print("   📁 Output Files Generated:")
print("      • sales_distributions.png")
print("      • sales_vs_advertising.png")
print("      • sales_correlation.png")
print("      • sales_budget_breakdown.png")
print("      • sales_actual_vs_predicted.png")
print("      • sales_feature_importance.png")
print("      • sales_model_comparison.png")
print("=" * 60)

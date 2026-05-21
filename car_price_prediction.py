# ============================================================
#   CAR PRICE PREDICTION — CodeAlpha Internship Task 3
#   Author: Your Name
#   Description: Predict car selling prices using ML
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("   CAR PRICE PREDICTION — CodeAlpha Task 3")
print("=" * 60)


# ─────────────────────────────────────────
# STEP 2: Load Dataset
# ─────────────────────────────────────────
df = pd.read_csv('car data.csv')

print("\n📂 Dataset Loaded Successfully!")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")


# ─────────────────────────────────────────
# STEP 3: Explore the Data (EDA)
# ─────────────────────────────────────────
print("\n─── First 5 Rows ───────────────────────────────────────")
print(df.head())

print("\n─── Dataset Info ────────────────────────────────────────")
print(df.info())

print("\n─── Missing Values ──────────────────────────────────────")
print(df.isnull().sum())

print("\n─── Statistical Summary ─────────────────────────────────")
print(df.describe())

print("\n─── Value Counts ────────────────────────────────────────")
print("Fuel Type:\n",    df['Fuel_Type'].value_counts())
print("\nTransmission:\n", df['Transmission'].value_counts())
print("\nSelling Type:\n", df['Selling_type'].value_counts())


# ─────────────────────────────────────────
# STEP 4: Feature Engineering
# ─────────────────────────────────────────
# Add Car Age feature
df['Car_Age'] = 2024 - df['Year']

print("\n⚙️  Feature Engineering Done!")
print(f"   Added 'Car_Age' column (2024 - Year)")


# ─────────────────────────────────────────
# STEP 5: Data Visualization
# ─────────────────────────────────────────
print("\n📊 Generating Visualizations...")

# Plot 1 — Selling Price Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['Selling_Price'], bins=30, color='#2196F3', kde=True, edgecolor='white')
plt.title("Distribution of Car Selling Prices", fontsize=14, fontweight='bold')
plt.xlabel("Selling Price (in Lakhs)")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("carprice_distribution.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: carprice_distribution.png")

# Plot 2 — Correlation Heatmap
plt.figure(figsize=(9, 7))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("carprice_correlation.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: carprice_correlation.png")

# Plot 3 — Fuel Type vs Selling Price
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x='Fuel_Type', y='Selling_Price',
            palette=['#4CAF50', '#2196F3', '#FF5722'])
plt.title("Selling Price by Fuel Type", fontsize=14, fontweight='bold')
plt.xlabel("Fuel Type")
plt.ylabel("Selling Price (in Lakhs)")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("carprice_fuel_type.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: carprice_fuel_type.png")

# Plot 4 — Car Age vs Selling Price
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x='Car_Age', y='Selling_Price',
                hue='Fuel_Type', palette=['#4CAF50', '#2196F3', '#FF5722'], alpha=0.7)
plt.title("Car Age vs Selling Price", fontsize=14, fontweight='bold')
plt.xlabel("Car Age (Years)")
plt.ylabel("Selling Price (in Lakhs)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("carprice_age_vs_price.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: carprice_age_vs_price.png")

# Plot 5 — Transmission vs Selling Price
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Transmission', y='Selling_Price',
            palette=['#9C27B0', '#FF9800'])
plt.title("Selling Price by Transmission Type", fontsize=14, fontweight='bold')
plt.xlabel("Transmission")
plt.ylabel("Selling Price (in Lakhs)")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("carprice_transmission.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: carprice_transmission.png")


# ─────────────────────────────────────────
# STEP 6: Prepare Data for ML
# ─────────────────────────────────────────
print("\n⚙️  Preparing Data for Machine Learning...")

# Encode categorical columns
le = LabelEncoder()
df['Fuel_Type']    = le.fit_transform(df['Fuel_Type'])
df['Selling_type'] = le.fit_transform(df['Selling_type'])
df['Transmission'] = le.fit_transform(df['Transmission'])

# Features and Target
X = df[['Present_Price', 'Driven_kms', 'Fuel_Type',
        'Selling_type', 'Transmission', 'Owner', 'Car_Age']]
y = df['Selling_Price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing  samples : {X_test.shape[0]}")


# ─────────────────────────────────────────
# STEP 7: Train Multiple Models
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
    print(f"   ✅ {name:<30} R² Score: {r2:.4f}  |  MAE: {mae:.2f}  |  RMSE: {rmse:.2f}")


# ─────────────────────────────────────────
# STEP 8: Best Model — Detailed Evaluation
# ─────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]["R2"])
best_preds = results[best_name]["predictions"]
best_model = results[best_name]["model"]

print(f"\n🏆 Best Model : {best_name}")
print(f"   R² Score   : {results[best_name]['R2']:.4f}  ({results[best_name]['R2']*100:.2f}% variance explained)")
print(f"   MAE        : {results[best_name]['MAE']:.2f} Lakhs")
print(f"   RMSE       : {results[best_name]['RMSE']:.2f} Lakhs")

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_preds, alpha=0.6, color='#2196F3', edgecolors='white', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linewidth=2, linestyle='--', label='Perfect Prediction')
plt.title(f"Actual vs Predicted Selling Price\n({best_name})", fontsize=13, fontweight='bold')
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("carprice_actual_vs_predicted.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: carprice_actual_vs_predicted.png")


# ─────────────────────────────────────────
# STEP 9: Feature Importance
# ─────────────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(9, 5))
    bars = plt.barh(importances.index, importances.values,
                    color=plt.cm.Blues(np.linspace(0.4, 0.9, len(importances))),
                    edgecolor='white')
    plt.title(f"Feature Importance — {best_name}", fontsize=13, fontweight='bold')
    plt.xlabel("Importance Score")
    for bar, val in zip(bars, importances.values):
        plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig("carprice_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("   ✅ Saved: carprice_feature_importance.png")


# ─────────────────────────────────────────
# STEP 10: Model Comparison Chart
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
plt.savefig("carprice_model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: carprice_model_comparison.png")


# ─────────────────────────────────────────
# STEP 11: Predict on a New Car Sample
# ─────────────────────────────────────────
print("\n🚗 Predicting Price for a New Car Sample...")

# Example: Present_Price=5.59, Driven_kms=27000, Fuel_Type=Petrol(1),
#          Selling_type=Dealer(0), Transmission=Manual(1), Owner=0, Car_Age=10
new_car = pd.DataFrame([[5.59, 27000, 1, 0, 1, 0, 10]],
                        columns=X.columns)
predicted_price = best_model.predict(new_car)[0]
print(f"   Input Features      : {new_car.values.tolist()[0]}")
print(f"   Predicted Car Price → ₹ {predicted_price:.2f} Lakhs")


# ─────────────────────────────────────────
# DONE
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("   ✅ ALL TASKS COMPLETED SUCCESSFULLY!")
print("   📁 Output Files Generated:")
print("      • carprice_distribution.png")
print("      • carprice_correlation.png")
print("      • carprice_fuel_type.png")
print("      • carprice_age_vs_price.png")
print("      • carprice_transmission.png")
print("      • carprice_actual_vs_predicted.png")
print("      • carprice_feature_importance.png")
print("      • carprice_model_comparison.png")
print("=" * 60)

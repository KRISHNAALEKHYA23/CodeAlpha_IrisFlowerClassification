# ============================================================
#   IRIS FLOWER CLASSIFICATION — CodeAlpha Internship Task 1
#   Author: Your Name
#   Description: Classify Iris flowers using Machine Learning
# ============================================================

# ─────────────────────────────────────────
# STEP 1: Import Libraries
# ─────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("=" * 55)
print("   IRIS FLOWER CLASSIFICATION — CodeAlpha Task 1")
print("=" * 55)


# ─────────────────────────────────────────
# STEP 2: Load Dataset
# ─────────────────────────────────────────
df = pd.read_csv("IRIS.csv")

# Drop the Id column — not needed for ML
df = df.drop('Id', axis=1)

print("\n📂 Dataset Loaded Successfully!")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")


# ─────────────────────────────────────────
# STEP 3: Explore the Data (EDA)
# ─────────────────────────────────────────
print("\n─── First 5 Rows ───────────────────────────────────")
print(df.head())

print("\n─── Dataset Info ────────────────────────────────────")
print(df.info())

print("\n─── Missing Values ──────────────────────────────────")
print(df.isnull().sum())

print("\n─── Species Count ───────────────────────────────────")
print(df['Species'].value_counts())

print("\n─── Statistical Summary ─────────────────────────────")
print(df.describe())


# ─────────────────────────────────────────
# STEP 4: Data Visualization
# ─────────────────────────────────────────
print("\n📊 Generating Visualizations...")

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
colors   = ['#4CAF50', '#2196F3', '#FF5722']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Iris Dataset — Exploratory Data Analysis", fontsize=16, fontweight='bold')

for ax, feature in zip(axes.flatten(), features):
    for i, species in enumerate(df['Species'].unique()):
        subset = df[df['Species'] == species][feature]
        ax.hist(subset, alpha=0.6, label=species, color=colors[i % 3], bins=15, edgecolor='white')
    ax.set_title(f"Distribution of {feature}", fontweight='bold')
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("iris_eda_distributions.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: iris_eda_distributions.png")

# Pairplot
print("   Generating Pairplot (this may take a moment)...")
pair = sns.pairplot(df, hue='Species', palette=['#4CAF50', '#2196F3', '#FF5722'], diag_kind='kde')
pair.fig.suptitle("Pairplot — All Feature Combinations by Species", y=1.02, fontsize=14, fontweight='bold')
plt.savefig("iris_pairplot.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: iris_pairplot.png")

# Correlation Heatmap
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="YlGnBu",
            linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("iris_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: iris_correlation_heatmap.png")


# ─────────────────────────────────────────
# STEP 5: Prepare Data for ML
# ─────────────────────────────────────────
print("\n⚙️  Preparing Data for Machine Learning...")

X = df.drop('Species', axis=1)
y = df['Species']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"   Classes: {list(le.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing  samples : {X_test.shape[0]}")


# ─────────────────────────────────────────
# STEP 6: Train Multiple Models & Compare
# ─────────────────────────────────────────
print("\n🤖 Training Models...")

models = {
    "K-Nearest Neighbors (KNN)"  : KNeighborsClassifier(n_neighbors=5),
    "Decision Tree"               : DecisionTreeClassifier(random_state=42),
    "Random Forest"               : RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {"model": model, "accuracy": accuracy, "predictions": y_pred}
    print(f"   ✅ {name:<35} Accuracy: {accuracy * 100:.2f}%")


# ─────────────────────────────────────────
# STEP 7: Best Model — Detailed Evaluation
# ─────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]["accuracy"])
best_model = results[best_name]["model"]
best_preds = results[best_name]["predictions"]

print(f"\n🏆 Best Model: {best_name}")
print(f"   Accuracy: {results[best_name]['accuracy'] * 100:.2f}%")

print("\n─── Classification Report ───────────────────────────")
print(classification_report(y_test, best_preds, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, best_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("iris_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: iris_confusion_matrix.png")


# ─────────────────────────────────────────
# STEP 8: Model Comparison Bar Chart
# ─────────────────────────────────────────
model_names = list(results.keys())
accuracies  = [results[m]["accuracy"] * 100 for m in model_names]
bar_colors  = ['#4CAF50' if m == best_name else '#90CAF9' for m in model_names]

plt.figure(figsize=(9, 5))
bars = plt.bar(model_names, accuracies, color=bar_colors, edgecolor='white', width=0.5)
plt.ylim(80, 105)
plt.title("Model Accuracy Comparison", fontsize=14, fontweight='bold')
plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{acc:.2f}%", ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig("iris_model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved: iris_model_comparison.png")


# ─────────────────────────────────────────
# STEP 9: Predict on a New Sample
# ─────────────────────────────────────────
print("\n🌸 Predicting Species for a New Sample...")

new_sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
)
prediction = best_model.predict(new_sample)
species    = le.inverse_transform(prediction)[0]

print(f"   Input Features : {new_sample.values.tolist()[0]}")
print(f"   Predicted Species → 🌺 {species.upper()}")


# ─────────────────────────────────────────
# DONE
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("   ✅ ALL TASKS COMPLETED SUCCESSFULLY!")
print("   📁 Output Files Generated:")
print("      • iris_eda_distributions.png")
print("      • iris_pairplot.png")
print("      • iris_correlation_heatmap.png")
print("      • iris_confusion_matrix.png")
print("      • iris_model_comparison.png")
print("=" * 55)

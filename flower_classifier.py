# ============================================================
# 🌸 Flower Species Classifier using Machine Learning
# Author: [Your Name]
# Description: Classifies Iris flower species using a
#              Decision Tree and Random Forest Classifier
# ============================================================

# ── Imports ──────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")


# ── Step 1: Load the Dataset ──────────────────────────────────
def load_data():
    print("\n📦 Loading Iris Dataset...")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map(dict(enumerate(iris.target_names)))
    print(f"   ✅ Loaded {len(df)} samples with {len(iris.feature_names)} features")
    return df, iris


# ── Step 2: Explore the Data ──────────────────────────────────
def explore_data(df):
    print("\n🔍 Dataset Overview:")
    print(df.head())
    print("\n📊 Species Distribution:")
    print(df['species_name'].value_counts())
    print("\n📈 Basic Statistics:")
    print(df.describe().round(2))


# ── Step 3: Visualize the Data ───────────────────────────────
def visualize_data(df, iris):
    print("\n🎨 Creating visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('🌸 Iris Dataset Exploration', fontsize=16, fontweight='bold')

    # Scatter plot: Sepal Length vs Petal Length
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    for i, species in enumerate(iris.target_names):
        subset = df[df['species'] == i]
        axes[0].scatter(
            subset['sepal length (cm)'],
            subset['petal length (cm)'],
            label=species, color=colors[i], alpha=0.7, s=60
        )
    axes[0].set_xlabel('Sepal Length (cm)')
    axes[0].set_ylabel('Petal Length (cm)')
    axes[0].set_title('Sepal vs Petal Length')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bar chart: Average measurements per species
    avg = df.groupby('species_name')[iris.feature_names].mean()
    avg.T.plot(kind='bar', ax=axes[1], color=colors, alpha=0.8)
    axes[1].set_title('Average Feature Values by Species')
    axes[1].set_xlabel('Feature')
    axes[1].set_ylabel('Value (cm)')
    axes[1].set_xticklabels(
        [f.replace(' (cm)', '') for f in iris.feature_names], rotation=30, ha='right'
    )
    axes[1].legend(title='Species')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('iris_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved → iris_visualization.png")


# ── Step 4: Train & Evaluate Models ──────────────────────────
def train_and_evaluate(df, iris):
    X = df[iris.feature_names]
    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n🔀 Train/Test Split: {len(X_train)} train | {len(X_test)} test")

    results = {}

    # Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_preds)
    results['Decision Tree'] = (dt_model, dt_preds, dt_acc)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    results['Random Forest'] = (rf_model, rf_preds, rf_acc)

    print("\n📋 Model Results:")
    for name, (model, preds, acc) in results.items():
        print(f"\n── {name} ──")
        print(f"   Accuracy: {acc * 100:.2f}%")
        print(classification_report(y_test, preds, target_names=iris.target_names))

    return results, X_test, y_test, iris


# ── Step 5: Confusion Matrix ──────────────────────────────────
def plot_confusion_matrices(results, y_test, iris):
    print("\n📊 Generating Confusion Matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')

    for ax, (name, (model, preds, acc)) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f'{name}\nAccuracy: {acc * 100:.1f}%')

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved → confusion_matrix.png")


# ── Step 6: Predict on New Input ─────────────────────────────
def predict_flower(model, iris):
    print("\n🌸 Predict a New Flower")
    print("   Enter measurements (in cm):")
    try:
        sepal_l = float(input("   Sepal Length: "))
        sepal_w = float(input("   Sepal Width:  "))
        petal_l = float(input("   Petal Length: "))
        petal_w = float(input("   Petal Width:  "))

        new_flower = [[sepal_l, sepal_w, petal_l, petal_w]]
        prediction = model.predict(new_flower)
        probabilities = model.predict_proba(new_flower)[0]

        species = iris.target_names[prediction[0]]
        confidence = max(probabilities) * 100

        print(f"\n   🌺 Predicted Species : {species.capitalize()}")
        print(f"   🎯 Confidence        : {confidence:.1f}%")

    except ValueError:
        print("   ❌ Invalid input. Please enter numeric values.")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  🌸 Flower Species Classifier")
    print("=" * 50)

    df, iris = load_data()
    explore_data(df)
    visualize_data(df, iris)
    results, X_test, y_test, iris = train_and_evaluate(df, iris)
    plot_confusion_matrices(results, y_test, iris)

    # Use Random Forest (best model) for prediction
    best_model = results['Random Forest'][0]
    predict_flower(best_model, iris)

    print("\n✅ Done! Check the generated PNG files for visualizations.")
    print("=" * 50)

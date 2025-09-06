import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, auc)

import joblib
sns.set(style="whitegrid")

CSV_PATH = "team15.csv"  
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

df = pd.read_csv(CSV_PATH)
print("Dataset loaded. Shape:", df.shape)
print(df.columns.tolist())
print(df.head())

target_col = pick_col(df, ["target", "cardio", "heart_disease", "has_disease", "label"])
if target_col is None:
    raise ValueError("Couldn't find a target column. Rename the label column to one of: "
                     "target, cardio, heart_disease, has_disease, label")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove id-like columns if present (common names)
id_cols = [c for c in ["id", "patient_id", "pid"] if c in df.columns]
numeric_cols = [c for c in numeric_cols if c not in id_cols + [target_col]]

cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

print("\nDetected:")
print(" Target:", target_col)
print(" Numeric features:", numeric_cols)
print(" Categorical features:", cat_cols)
print(" ID columns (ignored):", id_cols)

num_imputer = SimpleImputer(strategy="median")
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
if cat_cols:
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

label_encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].astype(str))
    label_encoders[c] = le

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]
X = df[feature_cols].values
y = df[target_col].values

print("\nFinal feature columns used:", feature_cols)
print("X shape:", X.shape, "y shape:", y.shape)

plt.figure(figsize=(6,4))
sns.countplot(x=df[target_col])
plt.title("Target Class Distribution")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "target_distribution.png"))
plt.close()

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"hist_{col}.png"))
    plt.close()

for col in numeric_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"box_{col}.png"))
    plt.close()

sample_cols = numeric_cols[:6]
if sample_cols:
    sns.pairplot(df[sample_cols + [target_col]], hue=target_col, corner=True, plot_kws={"alpha":0.6})
    plt.suptitle("Pairplot (sample numeric columns)", y=1.02)
    plt.savefig(os.path.join(FIG_DIR, "pairplot_sample.png"))
    plt.close()

corr = df[feature_cols + [target_col]].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation matrix (features + target)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "correlation_matrix.png"))
plt.close()

print("EDA plots saved to:", FIG_DIR)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Train/Test sizes:", X_train.shape[0], X_test.shape[0])

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, solver="liblinear"),
    "SVM": SVC(probability=True, kernel="rbf"),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining & evaluating: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")

    print(f" Accuracy: {acc:.4f}")
    print(f" Precision: {prec:.4f}")
    print(f" Recall: {rec:.4f}")
    print(f" F1-score: {f1:.4f}")
    print(f" ROC-AUC: {roc:.4f}")

    results[name] = {
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

    report = classification_report(y_test, y_pred, zero_division=0)
    with open(os.path.join(FIG_DIR, f"{name}_classification_report.txt"), "w") as f:
        f.write(report)

    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{name}_confusion_matrix.png"))
    plt.close()

summary = pd.DataFrame([
    {"model": name,
     "accuracy": info["accuracy"],
     "precision": info["precision"],
     "recall": info["recall"],
     "f1": info["f1"],
     "roc_auc": info["roc_auc"]}
    for name, info in results.items()
]).sort_values("f1", ascending=False).reset_index(drop=True)

print("\nModel comparison (sorted by F1):")
print(summary)
summary.to_csv(os.path.join(FIG_DIR, "model_comparison.csv"), index=False)

plt.figure(figsize=(8,5))
summary.set_index("model")[["accuracy", "precision", "recall", "f1"]].plot(kind="bar")
plt.title("Model comparison metrics")
plt.ylabel("Score")
plt.ylim(0,1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "model_comparison_metrics.png"))
plt.close()

plt.figure(figsize=(8,6))
for name, info in results.items():
    y_proba = info["y_proba"]
    if y_proba is None:
       
        model = info["model"]
        if hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            continue
    else:
        y_score = y_proba

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0,1], [0,1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "roc_curves.png"))
plt.close()

best = summary.iloc[0]["model"]
best_model = results[best]["model"]
joblib.dump(best_model, os.path.join(FIG_DIR, f"best_model_{best}.joblib"))
print(f"Best model by F1: {best} saved to {FIG_DIR}/best_model_{best}.joblib")

print("All figures and reports saved in folder:", FIG_DIR)

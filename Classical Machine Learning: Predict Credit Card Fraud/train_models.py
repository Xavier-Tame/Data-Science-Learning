import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc, precision_recall_curve
import seaborn as sns

GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

# Train, evaluate and graph models
def train_and_evaluate(X_train, y_train, X_test, y_test, dataset_name="Dataset"):
    print(f"\n===== Training on {dataset_name} =====")

    # Initialize models
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=10000, random_state=42, verbose=False
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, eval_metric='logloss', n_jobs=-1, random_state=42
        )
    }

    # ROC Curve
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"ROC Curve ({dataset_name})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    roc_path = os.path.join(GRAPH_DIR, f"{dataset_name}_ROC.png")
    plt.savefig(roc_path)
    plt.show()
    print(f"Saved ROC Curve: {roc_path}")

    # Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(recall, precision, label=name)

    plt.title(f"Precision-Recall Curve ({dataset_name})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    pr_path = os.path.join(GRAPH_DIR, f"{dataset_name}_PR.png")
    plt.savefig(pr_path)
    plt.show()
    print(f"Saved Precision-Recall Curve: {pr_path}")

    # Confusion Matrices
    for name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix ({name}) - {dataset_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_path = os.path.join(GRAPH_DIR, f"{dataset_name}_CM_{name}.png")
        plt.savefig(cm_path)
        plt.show()
        print(f"Saved Confusion Matrix for {name}: {cm_path}")


# Load datasets
scaled_train = pd.read_csv("creditcard_train_scaled.csv")
scaled_test = pd.read_csv("creditcard_test_scaled.csv")

unscaled_train = pd.read_csv("creditcard_train_unscaled.csv")
unscaled_test = pd.read_csv("creditcard_test_unscaled.csv")

datasets = {
    "Scaled": (
        scaled_train.drop('Class', axis=1), scaled_train['Class'],
        scaled_test.drop('Class', axis=1), scaled_test['Class']
    ),
    "Unscaled": (
        unscaled_train.drop('Class', axis=1), unscaled_train['Class'],
        unscaled_test.drop('Class', axis=1), unscaled_test['Class']
    )
}

# Train on both datasets
for name, (X_train, y_train, X_test, y_test) in datasets.items():
    train_and_evaluate(X_train, y_train, X_test, y_test, dataset_name=name)

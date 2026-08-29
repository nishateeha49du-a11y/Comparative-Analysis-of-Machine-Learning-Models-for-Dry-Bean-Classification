import os
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

SEEDS = [42, 10, 2026]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Dry_Bean_Dataset.csv"
RESULTS_PATH = BASE_DIR / "results" / "metrics" / "final_model_results.csv"


def build_models(seed):
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=seed)),
        ]),
        "Linear Discriminant Analysis": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearDiscriminantAnalysis()),
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Gaussian Naive Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(random_state=seed),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=seed)),
        ]),
        "Multi-layer Perceptron": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(max_iter=1000, random_state=seed)),
        ]),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()

    X = df.drop(columns=["Class"])
    y = df["Class"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    results = []

    for seed in SEEDS:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=seed,
            stratify=y,
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=seed,
            stratify=y_temp,
        )

        models = build_models(seed)

        validation_results = []

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)

            val_f1 = f1_score(
                y_val,
                y_val_pred,
                average="weighted",
                zero_division=0,
            )

            validation_results.append({
                "Model": model_name,
                "Validation F1": val_f1,
            })

        validation_df = pd.DataFrame(validation_results)
        validation_df = validation_df.sort_values("Validation F1", ascending=False)

        best_model_name = validation_df.iloc[0]["Model"]
        best_model = models[best_model_name]

        X_train_val = pd.concat([X_train, X_val], ignore_index=True)
        y_train_val = pd.concat(
            [pd.Series(y_train), pd.Series(y_val)],
            ignore_index=True,
        )

        best_model.fit(X_train_val, y_train_val)

        y_test_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(
            y_test,
            y_test_pred,
            average="weighted",
            zero_division=0,
        )
        recall = recall_score(
            y_test,
            y_test_pred,
            average="weighted",
            zero_division=0,
        )
        f1 = f1_score(
            y_test,
            y_test_pred,
            average="weighted",
            zero_division=0,
        )

        if hasattr(best_model, "predict_proba"):
            y_test_proba = best_model.predict_proba(X_test)
            roc_auc = roc_auc_score(
                y_test,
                y_test_proba,
                multi_class="ovr",
                average="weighted",
            )
        else:
            roc_auc = None

        results.append({
            "Random State": seed,
            "Model": best_model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC-AUC": roc_auc,
        })

        print(f"\nRandom State: {seed}")
        print(f"Selected Model: {best_model_name}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1: {f1:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_test_pred,
                target_names=label_encoder.classes_,
                zero_division=0,
            )
        )

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_test_pred))

    results_df = pd.DataFrame(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    print("\nFinal Results:")
    print(results_df.round(4))
    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
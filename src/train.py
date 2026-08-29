import os
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from models import get_models


SEEDS = [42, 100, 2026]

df = pd.read_csv("data/Dry_Bean_Dataset.csv")

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
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    models = get_models(random_state=seed)

    for model_name, model in models.items():

        if model_name in [
            "Logistic Regression",
            "Linear Discriminant Analysis",
            "K-Nearest Neighbors",
            "Support Vector Machine",
            "Multi-layer Perceptron"
        ]:
            X_train_model = X_train_scaled
            X_val_model = X_val_scaled
            X_test_model = X_test_scaled
        else:
            X_train_model = X_train
            X_val_model = X_val
            X_test_model = X_test

        start_time = time.time()

        model.fit(X_train_model, y_train)

        train_time = time.time() - start_time

        start_time = time.time()

        y_pred = model.predict(X_test_model)

        predict_time = time.time() - start_time

        y_proba = model.predict_proba(X_test_model)

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_proba,
            multi_class="ovr",
            average="weighted"
        )

        results.append({
            "Random State": seed,
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC-AUC": roc_auc,
            "Train Time": train_time,
            "Predict Time": predict_time
        })

        print(
            f"Seed: {seed} | "
            f"Model: {model_name} | "
            f"Accuracy: {accuracy:.4f} | "
            f"F1: {f1:.4f}"
        )


results_df = pd.DataFrame(results)

os.makedirs("results/metrics", exist_ok=True)

results_df.to_csv(
    "results/metrics/all_seed_results.csv",
    index=False
)

print("\nResults saved to:")
print("results/metrics/all_seed_results.csv")

print("\nResults:")
print(results_df)

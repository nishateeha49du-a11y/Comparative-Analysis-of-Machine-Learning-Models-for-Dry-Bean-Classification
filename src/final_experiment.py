import os
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


SEEDS = [42, 10, 2026]

df = pd.read_csv("data/Dry_Bean_Dataset.csv")
df = df.drop_duplicates()

X = df.drop(columns=["Class"])
y = df["Class"]

class_names = sorted(y.unique())
class_mapping = {
    class_name: index
    for index, class_name in enumerate(class_names)
}

y = y.map(class_mapping)

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

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=5000,
                random_state=seed
            ))
        ]),

        "Linear Discriminant Analysis": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearDiscriminantAnalysis())
        ]),

        "Decision Tree": DecisionTreeClassifier(
            random_state=seed
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=seed
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=seed
        ),

        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=5
            ))
        ]),

        "Gaussian Naive Bayes": GaussianNB(),

        "AdaBoost": AdaBoostClassifier(
            random_state=seed
        ),

        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                probability=True,
                random_state=seed
            ))
        ]),

        "Multi-layer Perceptron": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                max_iter=1000,
                random_state=seed
            ))
        ])
    }

    validation_results = []

    for model_name, model in models.items():

        start_time = time.time()

        model.fit(X_train, y_train)

        train_time = time.time() - start_time

        y_val_pred = model.predict(X_val)

        val_f1 = f1_score(
            y_val,
            y_val_pred,
            average="weighted",
            zero_division=0
        )

        validation_results.append({
            "Model": model_name,
            "Validation F1": val_f1
        })

    validation_results = pd.DataFrame(validation_results)

    validation_results = validation_results.sort_values(
        "Validation F1",
        ascending=False
    )

    best_model_name = validation_results.iloc[0]["Model"]

    X_train_val = pd.concat(
        [X_train, X_val],
        ignore_index=True
    )

    y_train_val = pd.concat(
        [
            pd.Series(y_train),
            pd.Series(y_val)
        ],
        ignore_index=True
    )

    final_model = models[best_model_name]

    start_time = time.time()

    final_model.fit(
        X_train_val,
        y_train_val
    )

    train_time = time.time() - start_time

    start_time = time.time()

    y_test_pred = final_model.predict(X_test)

    predict_time = time.time() - start_time

    y_test_proba = final_model.predict_proba(X_test)

    accuracy = accuracy_score(
        y_test,
        y_test_pred
    )

    precision = precision_score(
        y_test,
        y_test_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_test_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_test_pred,
        average="weighted",
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_test_proba,
        multi_class="ovr",
        average="weighted"
    )

    results.append({
        "Random State": seed,
        "Selected Model": best_model_name,
        "Validation F1": validation_results.iloc[0]["Validation F1"],
        "Test Accuracy": accuracy,
        "Test Precision": precision,
        "Test Recall": recall,
        "Test F1": f1,
        "Test ROC-AUC": roc_auc,
        "Train Time": train_time,
        "Predict Time": predict_time
    })

    print(f"\nRandom State: {seed}")
    print(f"Selected Model: {best_model_name}")
    print(f"Validation F1: {validation_results.iloc[0]['Validation F1']:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1: {f1:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")

    os.makedirs("results/models", exist_ok=True)

    joblib.dump(
        final_model,
        f"results/models/{best_model_name.replace(' ', '_')}_seed_{seed}.joblib"
    )


results_df = pd.DataFrame(results)

os.makedirs("results/metrics", exist_ok=True)

results_df.to_csv(
    "results/metrics/final_experiment_results.csv",
    index=False
)

print("\nFinal Experiment Results:")
print(results_df.round(4).to_string(index=False))

print(
    "\nSaved to: "
    "results/metrics/final_experiment_results.csv"
)


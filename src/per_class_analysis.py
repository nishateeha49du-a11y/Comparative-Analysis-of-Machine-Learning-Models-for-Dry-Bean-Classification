import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

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


def create_model(model_name, seed):
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

    return models[model_name]


df = pd.read_csv("data/Dry_Bean_Dataset.csv")
df = df.drop_duplicates()

X = df.drop(columns=["Class"])
y = df["Class"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

summary = pd.read_csv(
    "results/metrics/final_model_comparison.csv"
)

best_model_name = summary.iloc[0]["Model"]

print("Model selected for per-class analysis:")
print(best_model_name)

reports = []

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

    selected_model = create_model(best_model_name, seed)

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

    selected_model.fit(
        X_train_val,
        y_train_val
    )

    y_test_pred = selected_model.predict(X_test)

    report = classification_report(
        y_test,
        y_test_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    for class_name in label_encoder.classes_:
        reports.append({
            "Random State": seed,
            "Class": class_name,
            "Precision": report[class_name]["precision"],
            "Recall": report[class_name]["recall"],
            "F1": report[class_name]["f1-score"],
            "Support": report[class_name]["support"]
        })


reports_df = pd.DataFrame(reports)

class_summary = (
    reports_df
    .groupby("Class")
    .agg(
        Precision_Mean=("Precision", "mean"),
        Recall_Mean=("Recall", "mean"),
        F1_Mean=("F1", "mean"),
        F1_Std=("F1", "std"),
        Support=("Support", "mean")
    )
    .reset_index()
)

class_summary = class_summary.round(4)

os.makedirs("results/metrics", exist_ok=True)
os.makedirs("figures", exist_ok=True)

class_summary.to_csv(
    "results/metrics/per_class_summary.csv",
    index=False
)

print("\nPer-Class Performance:")
print(class_summary.to_string(index=False))

plt.figure(figsize=(10, 6))

sns.barplot(
    data=class_summary,
    x="F1_Mean",
    y="Class"
)

plt.title(
    f"Per-Class F1 Score - {best_model_name}"
)

plt.xlabel("Mean F1 Score")
plt.ylabel("Bean Class")
plt.xlim(0, 1)

plt.tight_layout()

plt.savefig(
    "figures/per_class_f1.png",
    dpi=300
)

plt.close()

print(
    "\nSaved to: "
    "results/metrics/per_class_summary.csv"
)

print(
    "Figure saved to: "
    "figures/per_class_f1.png"
)

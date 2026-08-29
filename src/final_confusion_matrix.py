import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

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

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

summary = pd.read_csv(
    "results/metrics/final_summary.csv"
)

best_model_name = summary.iloc[0]["Model"]

print("Best Model:", best_model_name)

os.makedirs("figures", exist_ok=True)


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

    model = create_model(best_model_name, seed)

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

    model.fit(X_train_val, y_train_val)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )

    plt.title(
        f"{best_model_name} Confusion Matrix - Seed {seed}"
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.tight_layout()

    filename = (
        f"figures/confusion_matrix_seed_{seed}.png"
    )

    plt.savefig(
        filename,
        dpi=300
    )

    plt.close()

    print(f"Confusion matrix saved for seed {seed}")
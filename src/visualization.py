import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


SEED = 42

df = pd.read_csv("data/Dry_Bean_Dataset.csv")

df = df.drop_duplicates()

X = df.drop(columns=["Class"])
y = df["Class"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=SEED
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

cm = confusion_matrix(y_test, y_pred)

os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()

plt.savefig(
    "figures/random_forest_confusion_matrix.png",
    dpi=300
)

plt.close()
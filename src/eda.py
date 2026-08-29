import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/Dry_Bean_Dataset.csv")
print("Dataset Shape:", df.shape)
print("Column Names:",df.columns.tolist())
print("Firt 5 rows:, df.head())")
print("Data type",df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nTotal Missing Values:")
print(df.isnull().sum().sum())
print("\nDuplicate Rows:", df.duplicated().sum())
print("\nDescriptive Statistics:")
print(df.describe())
print("\nClass Distribution (%):")
print(df["Class"].value_counts(normalize=True) * 100)
class_counts = df["Class"].value_counts()

plt.figure(figsize=(10, 6))
class_counts.plot(kind="bar")

plt.title("Distribution of Bean Classes")
plt.xlabel("Bean Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("figures/bean_class_distribution.png", dpi=300)
plt.close()

df.hist(figsize=(16, 12), bins=30)

plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.tight_layout()
plt.savefig("figures/numerical_features_distribution.png", dpi=300)
plt.close()



plt.figure(figsize=(14, 10))

correlation = df.drop(columns=["Class"]).corr()

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("figures/feature_correlation_matrix.png", dpi=300)
plt.close()

features = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter"
]

for feature in features:
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="Class",
        y=feature
    )

    plt.title(f"{feature} by Bean Class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"figures/{feature.lower()}_by_bean_class.png", dpi=300)
    plt.close()


    from sklearn.ensemble import RandomForestClassifier

# Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X, y)

# Feature importance
importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)
plt.figure(figsize=(10, 7))

importance.sort_values().plot(kind="barh")

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig("figures/random_forest_feature_importance.png", dpi=300)
plt.close()








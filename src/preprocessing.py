import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

SEEDS = [42, 10, 2026]

df = pd.read_csv("data/Dry_Bean_Dataset.csv")

print("Original shape:", df.shape)

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

X = df.drop(columns=["Class"])
y = df["Class"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nEncoded classes:")
for label, class_name in enumerate(label_encoder.classes_):
    print(label, "=", class_name)

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

    print(f"\nRandom State: {seed}")

    print("Training:", X_train_scaled.shape)
    print("Validation:", X_val_scaled.shape)
    print("Testing:", X_test_scaled.shape)

    print("Training class distribution:")
    print(pd.Series(y_train).value_counts().sort_index())
import pandas as pd

results = pd.read_csv("results/metrics/all_seed_results.csv")

summary = (
    results
    .groupby("Model")
    .agg(
        Accuracy_Mean=("Accuracy", "mean"),
        Accuracy_Std=("Accuracy", "std"),
        Precision_Mean=("Precision", "mean"),
        Recall_Mean=("Recall", "mean"),
        F1_Mean=("F1", "mean"),
        F1_Std=("F1", "std"),
        ROC_AUC_Mean=("ROC-AUC", "mean"),
        Train_Time_Mean=("Train Time", "mean"),
        Predict_Time_Mean=("Predict Time", "mean")
    )
    .reset_index()
)

summary = summary.sort_values(
    by="F1_Mean",
    ascending=False
)

summary = summary.round(4)

summary.to_csv(
    "results/metrics/seed_comparison_summary.csv",
    index=False
)

print("Seed comparison summary:")
print(summary)

print("\nSummary saved to:")
print("results/metrics/seed_comparison_summary.csv")
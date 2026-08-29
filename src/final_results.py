import pandas as pd

results = pd.read_csv(
    "results/metrics/final_experiment_results.csv"
)

summary = (
    results
    .groupby("Selected Model")
    .agg(
        Seeds=("Random State", "count"),
        Accuracy_Mean=("Test Accuracy", "mean"),
        Accuracy_Std=("Test Accuracy", "std"),
        Precision_Mean=("Test Precision", "mean"),
        Recall_Mean=("Test Recall", "mean"),
        F1_Mean=("Test F1", "mean"),
        F1_Std=("Test F1", "std"),
        ROC_AUC_Mean=("Test ROC-AUC", "mean"),
        Train_Time_Mean=("Train Time", "mean"),
        Predict_Time_Mean=("Predict Time", "mean")
    )
    .reset_index()
)

summary = summary.sort_values(
    "F1_Mean",
    ascending=False
)

summary = summary.round(4)

print("\nFinal Performance Summary")
print(summary.to_string(index=False))

summary.to_csv(
    "results/metrics/final_performance_summary.csv",
    index=False
)

print(
    "\nSaved to: "
    "results/metrics/final_performance_summary.csv"
)
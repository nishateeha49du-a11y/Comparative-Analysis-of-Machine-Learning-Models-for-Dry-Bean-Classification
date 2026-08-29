import pandas as pd

results = pd.read_csv(
    "results/metrics/all_seed_results.csv"
)

summary = (
    results
    .groupby("Model")
    .agg(
        Accuracy_Mean=("Accuracy", "mean"),
        Accuracy_Std=("Accuracy", "std"),
        Precision_Mean=("Precision", "mean"),
        Precision_Std=("Precision", "std"),
        Recall_Mean=("Recall", "mean"),
        Recall_Std=("Recall", "std"),
        F1_Mean=("F1", "mean"),
        F1_Std=("F1", "std"),
        ROC_AUC_Mean=("ROC-AUC", "mean"),
        ROC_AUC_Std=("ROC-AUC", "std"),
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

print("\nOverall Model Comparison")
print(
    summary.to_string(index=False)
)

summary.to_csv(
    "results/metrics/overall_model_comparison.csv",
    index=False
)

print(
    "\nSaved to: "
    "results/metrics/overall_model_comparison.csv"
)

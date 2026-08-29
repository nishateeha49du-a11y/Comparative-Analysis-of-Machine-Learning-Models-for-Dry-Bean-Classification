import pandas as pd

summary = pd.read_csv(
    "results/metrics/overall_model_comparison.csv"
)

report = summary[
    [
        "Model",
        "Accuracy_Mean",
        "Accuracy_Std",
        "Precision_Mean",
        "Recall_Mean",
        "F1_Mean",
        "F1_Std",
        "ROC_AUC_Mean",
        "ROC_AUC_Std"
    ]
].copy()

report = report.rename(
    columns={
        "Accuracy_Mean": "Accuracy",
        "Accuracy_Std": "Accuracy Std",
        "Precision_Mean": "Precision",
        "Recall_Mean": "Recall",
        "F1_Mean": "F1",
        "F1_Std": "F1 Std",
        "ROC_AUC_Mean": "ROC-AUC",
        "ROC_AUC_Std": "ROC-AUC Std"
    }
)

report = report.round(4)

print("\nFinal Model Comparison Table")
print(report.to_string(index=False))

report.to_csv(
    "results/metrics/report_results.csv",
    index=False
)

print(
    "\nSaved to: "
    "results/metrics/report_results.csv"
)

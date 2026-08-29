import pandas as pd

summary = pd.read_csv(
    "results/metrics/overall_model_comparison.csv"
)

best_model = summary.iloc[0]

print("\nOverall Best Model")
print("------------------")
print("Model:", best_model["Model"])
print("Mean Accuracy:", best_model["Accuracy_Mean"])
print("Accuracy Std:", best_model["Accuracy_Std"])
print("Mean Precision:", best_model["Precision_Mean"])
print("Mean Recall:", best_model["Recall_Mean"])
print("Mean F1:", best_model["F1_Mean"])
print("F1 Std:", best_model["F1_Std"])
print("Mean ROC-AUC:", best_model["ROC_AUC_Mean"])
print("ROC-AUC Std:", best_model["ROC_AUC_Std"])
print("Mean Training Time:", best_model["Train_Time_Mean"])
print("Mean Prediction Time:", best_model["Predict_Time_Mean"])

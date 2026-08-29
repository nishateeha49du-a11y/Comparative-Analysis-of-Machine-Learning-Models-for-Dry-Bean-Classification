import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

results = pd.read_csv("results/metrics/all_seed_results.csv")

summary = (
    results
    .groupby("Model")
    .agg(
        Accuracy=("Accuracy", "mean"),
        Precision=("Precision", "mean"),
        Recall=("Recall", "mean"),
        F1=("F1", "mean"),
        ROC_AUC=("ROC-AUC", "mean")
    )
    .reset_index()
)

summary = summary.sort_values("F1", ascending=False)

print("\nModel Performance Summary:")
print(summary.round(4))

plt.figure(figsize=(12, 7))

sns.barplot(
    data=summary,
    x="F1",
    y="Model"
)

plt.title("Average F1 Score Across Random States")
plt.xlabel("F1 Score")
plt.ylabel("Model")
plt.xlim(0, 1)
plt.tight_layout()

plt.savefig(
    "figures/model_f1_comparison.png",
    dpi=300
)

plt.close()
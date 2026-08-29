import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results = pd.read_csv(
    "results/metrics/final_model_results.csv"
)

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC"
]

for metric in metrics:

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=results,
        x="Random State",
        y=metric,
        hue="Model"
    )

    plt.title(
        f"{metric} Across Random States"
    )

    plt.xlabel("Random State")
    plt.ylabel(metric)

    plt.ylim(0, 1)

    plt.tight_layout()

    filename = (
        f"figures/{metric.lower().replace('-', '_')}_by_seed.png"
    )

    plt.savefig(
        filename,
        dpi=300
    )

    plt.show()
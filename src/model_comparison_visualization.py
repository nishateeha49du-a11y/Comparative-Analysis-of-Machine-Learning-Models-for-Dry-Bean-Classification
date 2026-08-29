import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results = pd.read_csv(
    "results/metrics/all_seed_results.csv"
)

plt.figure(figsize=(12, 7))

sns.barplot(
    data=results,
    x="Model",
    y="F1",
    hue="Random State"
)

plt.title("F1 Score Comparison Across Random States")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig(
    "figures/all_models_f1_comparison.png",
    dpi=300
)

plt.show()

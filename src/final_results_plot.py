import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

summary = pd.read_csv(
    "results/metrics/final_summary.csv"
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=summary,
    x="F1_Mean",
    y="Model"
)

plt.title("Final Model Performance")
plt.xlabel("Mean Test F1 Score")
plt.ylabel("Model")
plt.xlim(0, 1)
plt.tight_layout()

plt.savefig(
    "figures/final_model_performance.png",
    dpi=300
)

plt.close()
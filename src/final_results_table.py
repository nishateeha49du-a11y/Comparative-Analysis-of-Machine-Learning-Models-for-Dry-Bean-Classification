import pandas as pd

results = pd.read_csv(
    "results/metrics/final_model_results.csv"
)

results = results.round(4)

print("\nFinal Experiment Results")

print(results.to_string(index=False))

results.to_csv(
    "results/metrics/final_results_table.csv",
    index=False
)

print(
    "\nSaved to: "
    "results/metrics/final_results_table.csv"
)
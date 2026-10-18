
import pandas as pd

df = pd.read_csv("results/metrics.csv")

# Select one fixed configuration for a fair comparison.
# Change these values if you want to examine another setting.
selected_top_k = 40
selected_top_p = 0.9

subset = df[
    (df["top_k"] == selected_top_k)
    & (df["top_p"] == selected_top_p)
]

summary = (
    subset.groupby(["model", "temperature"], as_index=False)
    .agg(
        mean_consistency=("exact_consistency_rate", "mean"),
        mean_text_similarity=("mean_pairwise_text_similarity", "mean"),
        mean_json_validity=("json_valid_rate", "mean"),
    )
)

print("\nTEMPERATURE COMPARISON")
print(f"Fixed Top-K = {selected_top_k}, Top-P = {selected_top_p}")
print(summary.to_string(index=False))

summary.to_csv("results/temperature_comparison.csv", index=False)
print("\nSaved: results/temperature_comparison.csv")

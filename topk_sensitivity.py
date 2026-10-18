
import pandas as pd
from difflib import SequenceMatcher

# Load experiment results
df = pd.read_csv("results/responses.csv")

# Keep Top-P and Temperature fixed
subset = df[
    (df["temperature"] == 0.7) &
    (df["top_p"] == 0.9)
].copy()

top_k_values = sorted(subset["top_k"].dropna().unique())

print("\nRESPONSE CHANGE ACROSS TOP-K VALUES")
print("Fixed Temperature = 0.7 | Fixed Top-P = 0.9")

for model in sorted(subset["model"].dropna().unique()):
    model_data = subset[subset["model"] == model]

    # Select one response per Top-K value (first repeat)
    representatives = (
        model_data.sort_values("repeat")
        .drop_duplicates(subset=["top_k"])
        .sort_values("top_k")
    )

    print(f"\nModel: {model}")
    previous = None
    previous_top_k = None

    for _, row in representatives.iterrows():
        response = str(row["response"])
        top_k = row["top_k"]

        if pd.isna(row["response"]) or not response.strip():
            print(f"Top-K {top_k}: no response available")
            continue

        if previous is None:
            print(f"Top-K {top_k}: baseline")
        else:
            similarity = SequenceMatcher(
                None, previous, response
            ).ratio()

            print(
                f"Top-K {previous_top_k} -> {top_k}: "
                f"similarity = {similarity:.3f}"
            )

        previous = response
        previous_top_k = top_k

print("\nHigher similarity means more similar wording.")
print("This is a text comparison, not an accuracy score.")

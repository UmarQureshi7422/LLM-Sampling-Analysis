
import pandas as pd
from difflib import SequenceMatcher

# Load experiment results
df = pd.read_csv("results/responses.csv")

# Keep Temperature and Top-K fixed
subset = df[
    (df["temperature"] == 0.7) &
    (df["top_k"] == 40)
].copy()

print("\nRESPONSE CHANGE ACROSS TOP-P VALUES")
print("Fixed Temperature = 0.7 | Fixed Top-K = 40")

for model in sorted(subset["model"].dropna().unique()):
    model_data = subset[subset["model"] == model]

    # Select one response per Top-P value (first repeat)
    representatives = (
        model_data.sort_values("repeat")
        .drop_duplicates(subset=["top_p"])
        .sort_values("top_p")
    )

    print(f"\nModel: {model}")
    previous = None
    previous_top_p = None

    for _, row in representatives.iterrows():
        response = str(row["response"])
        top_p = row["top_p"]

        if pd.isna(row["response"]) or not response.strip():
            print(f"Top-P {top_p}: no response available")
            continue

        if previous is None:
            print(f"Top-P {top_p}: baseline")
        else:
            similarity = SequenceMatcher(
                None, previous, response
            ).ratio()

            print(
                f"Top-P {previous_top_p} -> {top_p}: "
                f"similarity = {similarity:.3f}"
            )

        previous = response
        previous_top_p = top_p

print("\nHigher similarity means more similar wording.")
print("This is a text comparison, not an accuracy score.")

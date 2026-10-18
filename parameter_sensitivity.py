
import pandas as pd
from difflib import SequenceMatcher

df = pd.read_csv("results/responses.csv")

# Compare responses while keeping Top-K and Top-P fixed.
subset = df[
    (df["top_k"] == 40)
    & (df["top_p"] == 0.9)
].copy()

print("\nRESPONSE CHANGE ACROSS TEMPERATURES")

for model in sorted(subset["model"].dropna().unique()):
    model_data = subset[subset["model"] == model]

    # Select one response per temperature for this comparison.
    representatives = (
        model_data.sort_values("repeat")
        .drop_duplicates(subset=["temperature"])
        .sort_values("temperature")
    )

    previous = None

    print(f"\nModel: {model}")

    for _, row in representatives.iterrows():
        response = str(row["response"])
        temperature = row["temperature"]

        if previous is None:
            print(f"Temperature {temperature}: baseline")
        else:
            similarity = SequenceMatcher(
                None,
                previous,
                response,
            ).ratio()

            print(
                f"Temperature {temperature}: "
                f"similarity to previous temperature = "
                f"{similarity:.3f}"
            )

        previous = response

print("\nHigher similarity means more similar wording.")
print("This is a text comparison, not a factual accuracy score.")

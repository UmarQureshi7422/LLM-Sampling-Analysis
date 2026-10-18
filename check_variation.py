
import pandas as pd

df = pd.read_csv("results/responses.csv")

print("Available columns:")
print(df.columns.tolist())

print("\nResponse variation by model and temperature:")

for model in df["model"].dropna().unique():
    for temp in sorted(df["temperature"].dropna().unique()):
        subset = df[
            (df["model"] == model)
            & (df["temperature"] == temp)
            & (df["top_k"] == 40)
            & (df["top_p"] == 0.9)
        ]

        if subset.empty:
            continue

        response_col = "response"
        unique_count = subset[response_col].nunique()

        print(
            f"\nModel={model}, Temperature={temp}, "
            f"Responses={len(subset)}, "
            f"Unique responses={unique_count}"
        )

        for _, row in subset.iterrows():
            print(str(row[response_col])[:250].replace("\n", " "))

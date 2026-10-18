
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

results_dir = Path("results")
metrics_file = results_dir / "metrics.csv"
results_dir.mkdir(exist_ok=True)

if not metrics_file.exists():
    raise FileNotFoundError(
        "results/metrics.csv not found. Run main.py first."
    )

df = pd.read_csv(metrics_file)

if df.empty:
    raise ValueError("metrics.csv has no metric rows.")

# Plot 1: Response consistency by temperature
consistency = (
    df.groupby(["model", "temperature"], as_index=False)
    ["exact_consistency_rate"]
    .mean()
)

for model in consistency["model"].dropna().unique():
    subset = consistency[consistency["model"] == model].sort_values("temperature")
    plt.plot(
        subset["temperature"],
        subset["exact_consistency_rate"],
        marker="o",
        label=model,
    )

plt.xlabel("Temperature")
plt.ylabel("Mean exact consistency rate")
plt.title("Response Consistency vs Temperature")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(results_dir / "consistency_vs_temperature.png", dpi=160)
plt.close()

# Plot 2: JSON validity by model and temperature
json_rates = (
    df.groupby(["model", "temperature"], as_index=False)["json_valid_rate"]
    .mean()
)

for model in json_rates["model"].dropna().unique():
    subset = json_rates[json_rates["model"] == model].sort_values("temperature")
    plt.plot(
        subset["temperature"],
        subset["json_valid_rate"],
        marker="o",
        label=model,
    )

plt.xlabel("Temperature")
plt.ylabel("Mean JSON validity rate")
plt.title("JSON Output Validity vs Temperature")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(results_dir / "json_validity_vs_temperature.png", dpi=160)
plt.close()

print("Charts saved:")
print(results_dir / "consistency_vs_temperature.png")
print(results_dir / "json_validity_vs_temperature.png")


"""Run controlled Ollama sampling experiments and save responses/metrics."""
import argparse
import itertools
import json
import time
from pathlib import Path
import pandas as pd
import requests

from evaluation import evaluate_run_group

DEFAULT_MODELS = ["llama3", "qwen2.5"]
DEFAULT_TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
DEFAULT_TOP_K = [10, 40, 100]
DEFAULT_TOP_P = [0.5, 0.9, 1.0]


def parse_csv_values(value, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def load_prompts(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("prompts.json must contain a non-empty JSON list.")
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "id" not in item or "prompt" not in item:
            raise ValueError(f"Prompt at index {i} needs 'id' and 'prompt' fields.")
    return data


def generate(base_url, model, prompt, temperature, top_k, top_p, timeout):
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "seed": 42
        }
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    return result.get("response", ""), result


def main():
    parser = argparse.ArgumentParser(description="Run an Ollama sampling-parameter experiment.")
    parser.add_argument("--models", default="llama3,qwen2.5", help="Comma-separated Ollama model tags")
    parser.add_argument("--prompts", default="prompts.json")
    parser.add_argument("--temperatures", default="0,0.3,0.7,1.0")
    parser.add_argument("--top-k", default="10,40,100")
    parser.add_argument("--top-p", default="0.5,0.9,1.0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    prompts = load_prompts(args.prompts)
    models = parse_csv_values(args.models, str)
    temperatures = parse_csv_values(args.temperatures, float)
    top_ks = parse_csv_values(args.top_k, int)
    top_ps = parse_csv_values(args.top_p, float)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_responses.jsonl"
    rows = []
    total = len(models) * len(prompts) * len(temperatures) * len(top_ks) * len(top_ps) * args.repeats
    completed = 0

    with raw_path.open("w", encoding="utf-8") as raw_file:
        for model, prompt_item, temp, top_k, top_p in itertools.product(
            models, prompts, temperatures, top_ks, top_ps
        ):
            for repeat in range(1, args.repeats + 1):
                completed += 1
                record = {
                    "model": model,
                    "prompt_id": prompt_item["id"],
                    "prompt": prompt_item["prompt"],
                    "temperature": temp,
                    "top_k": top_k,
                    "top_p": top_p,
                    "repeat": repeat,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
                }
                try:
                    answer, metadata = generate(
                        args.base_url, model, prompt_item["prompt"],
                        temp, top_k, top_p, args.timeout
                    )
                    record["response"] = answer
                    record["error"] = None
                    record["total_duration_ns"] = metadata.get("total_duration")
                    record["eval_count"] = metadata.get("eval_count")
                    record["prompt_eval_count"] = metadata.get("prompt_eval_count")
                except Exception as exc:
                    record["response"] = ""
                    record["error"] = f"{type(exc).__name__}: {exc}"
                raw_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                raw_file.flush()
                rows.append(record)
                print(f"[{completed}/{total}] {model} prompt={prompt_item['id']} "
                      f"T={temp} K={top_k} P={top_p} repeat={repeat} "
                      f"{'ERROR' if record['error'] else 'OK'}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "responses.csv", index=False)
    metrics = evaluate_run_group(rows, prompts)
    pd.DataFrame(metrics).to_csv(out_dir / "metrics.csv", index=False)

    # Simple plots are optional so a successful run still saves data if plotting fails.
    try:
        import matplotlib.pyplot as plt
        successful = df[df["error"].isna()].copy()
        if not successful.empty:
            grouped = successful.groupby(["model", "temperature"])["response"].nunique().reset_index()
            grouped.rename(columns={"response": "unique_responses"}, inplace=True)
            for model in grouped["model"].unique():
                subset = grouped[grouped["model"] == model]
                plt.figure()
                plt.plot(subset["temperature"], subset["unique_responses"], marker="o")
                plt.xlabel("Temperature")
                plt.ylabel("Unique responses (aggregated)")
                plt.title(f"Response variation by temperature — {model}")
                plt.tight_layout()
                safe_model = "".join(c if c.isalnum() or c in "-_" else "_" for c in model)
                plt.savefig(out_dir / f"variation_{safe_model}.png", dpi=160)
                plt.close()
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")

    print(f"\nDone. Raw data: {raw_path}")
    print(f"Responses: {out_dir / 'responses.csv'}")
    print(f"Metrics: {out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()

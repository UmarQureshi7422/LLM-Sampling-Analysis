
# LLM Sampling Analysis

A reproducible experiment framework for examining how Temperature, Top-K, and Top-P settings affect outputs from local language models served by [Ollama](https://ollama.com/).

## Research question

How do sampling settings affect response variation, consistency, and output structure across Llama 3 and Qwen 2.5? Hallucination is discussed cautiously: automated text-only heuristics cannot reliably determine whether a response is factually false.

## Models and tools

- Ollama models: `llama3` and `qwen2.5` (download tags supported by your Ollama installation)
- Python 3.10+
- Requests, Pandas, NumPy, Matplotlib

## Setup

1. Install Ollama and start it.
2. Download the models:
   ```bash
   ollama pull llama3
   ollama pull qwen2.5
   ```
3. Create and activate a virtual environment (recommended).
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run a small test first

The full grid can involve many calls. Start small to check that both model tags work:

```bash
python main.py --models llama3 --temperatures 0,0.7 --top-k 40 --top-p 0.9 --repeats 2
```

Run the default experiment:

```bash
python main.py
```

Default grid: 2 models × 5 prompts × 4 temperatures × 3 Top-K values × 3 Top-P values × 3 repeats = **1,080 generation requests**. Adjust arguments to fit your hardware and time budget.

Ollama defaults to `http://localhost:11434`. Override with `--base-url` if needed.

## Outputs

Generated under `results/`:
- `raw_responses.jsonl`: one record per request, including errors and timing metadata
- `responses.csv`: tabular raw results
- `metrics.csv`: aggregated metrics per model/prompt/parameter combination
- `variation_<model>.png`: simple response-variation plots

## Metrics and limitations

- **Exact consistency:** measures how often repeated normalized responses match.
- **Pairwise text similarity:** rough measure of textual similarity between repeats.
- **JSON validity:** whether the entire response parses as JSON.
- **Reference lexical similarity:** optional string similarity against a reference answer. This is **not** a factuality or hallucination score.
- **Hallucination:** requires reference-backed, task-specific factual verification or human review. Do not claim that lexical similarity alone measures hallucination.

A fixed seed is passed for repeatability, but this does not guarantee identical outputs across all models, hardware, Ollama versions, or sampling settings. In particular, a temperature of zero can make outputs more deterministic, but exact repeatability should be measured rather than assumed.

## Suggested report structure

1. Abstract and research question
2. Background on sampling parameters
3. Experimental design and prompt dataset
4. Metrics and limitations
5. Results (tables and plots)
6. Discussion and threats to validity
7. Conclusion

## Repository contents

```text
.
├── main.py
├── evaluation.py
├── prompts.json
├── requirements.txt
├── README.md
├── .gitignore
├── results/
└── report/
```

## Notes

Use the same prompt set and experiment grid for each model. Report failed requests rather than silently dropping them. Before submitting, include actual experiment results and a PDF report based on the runs you performed; do not fabricate results.

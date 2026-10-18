import json
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="LLM Sampling Analysis",
    page_icon="🧪",
    layout="wide",
)

RESULTS = Path("results")
METRICS_FILE = RESULTS / "metrics.csv"
RESPONSES_FILE = RESULTS / "responses.csv"

st.title("🧪 LLM Sampling Analysis")
st.caption(
    "Explore how Temperature, Top-K, and Top-P affect "
    "local LLM outputs: Llama 3 vs Qwen 2.5."
)

# --------------------------------------------------
# Load data
# --------------------------------------------------
if not METRICS_FILE.exists():
    st.warning("No metrics found. Run your experiment first.")
    st.code(
        "python main.py --models llama3,qwen2.5 "
        "--temperatures 0,0.3,0.7,1.0 "
        "--top-k 10,40,100 --top-p 0.5,0.9,1.0 --repeats 2",
        language="powershell",
    )
    st.stop()

metrics = pd.read_csv(METRICS_FILE)

if RESPONSES_FILE.exists():
    responses = pd.read_csv(RESPONSES_FILE)
else:
    responses = pd.DataFrame()

# Convert numeric columns safely
for frame in (metrics, responses):
    for column in ("temperature", "top_k", "top_p", "repeat"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column], errors="coerce"
            )

# Remove failed or empty responses from text comparisons
if not responses.empty and "response" in responses.columns:
    responses["response"] = responses["response"].fillna("").astype(str)

    if "error" in responses.columns:
        responses["error"] = responses["error"].fillna("").astype(str)
        responses = responses[
            (responses["error"] == "")
            & (responses["response"].str.strip() != "")
        ].copy()
    else:
        responses = responses[
            responses["response"].str.strip() != ""
        ].copy()

# --------------------------------------------------
# Word-count compliance (computed from raw responses)
# --------------------------------------------------
WORD_COUNT_TARGET = 80
prompts_file = Path("prompts.json")
if prompts_file.exists():
    try:
        prompt_definitions = json.loads(prompts_file.read_text(encoding="utf-8"))
        target_by_id = {
            str(item.get("id")): int(item["word_count"])
            for item in prompt_definitions
            if item.get("word_count") is not None
        }
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        target_by_id = {}
else:
    target_by_id = {}

if not responses.empty and "response" in responses.columns:
    responses["word_count"] = responses["response"].apply(
        lambda value: len(str(value or "").split())
    )
    if "prompt_id" in responses.columns:
        responses["word_count_target"] = responses["prompt_id"].astype(str).map(
            target_by_id
        ).fillna(WORD_COUNT_TARGET).astype(int)
    else:
        responses["word_count_target"] = WORD_COUNT_TARGET
    responses["meets_word_count"] = (
        responses["word_count"] == responses["word_count_target"]
    )

# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.header("Experiment filters")

available_models = sorted(
    metrics["model"].dropna().unique().tolist()
)
selected_models = st.sidebar.multiselect(
    "Models",
    available_models,
    default=available_models,
)

available_temperatures = sorted(
    metrics["temperature"].dropna().unique().tolist()
)
selected_temperatures = st.sidebar.multiselect(
    "Temperatures",
    available_temperatures,
    default=available_temperatures,
)

filtered = metrics[
    metrics["model"].isin(selected_models)
    & metrics["temperature"].isin(selected_temperatures)
].copy()

if filtered.empty:
    st.info("Select at least one model and temperature.")
    st.stop()

# --------------------------------------------------
# Summary cards
# --------------------------------------------------
st.subheader("Experiment overview")

c1, c2, c3 = st.columns(3)

c1.metric("Configurations", len(filtered))

if "exact_consistency_rate" in filtered.columns:
    c2.metric(
        "Mean repeat consistency",
        f"{filtered['exact_consistency_rate'].mean():.1%}",
    )
else:
    c2.metric("Mean repeat consistency", "N/A")

if not responses.empty:
    c3.metric("Successful responses", len(responses))
else:
    c3.metric("Successful responses", "N/A")

st.info(
    "Repeat consistency measures agreement between repeated runs "
    "of the same configuration. Sensitivity below compares outputs "
    "between different parameter settings."
)

# --------------------------------------------------
# Overview charts
# --------------------------------------------------
overview_tab, sensitivity_tab, data_tab = st.tabs(
    ["📊 Overview", "🔬 Parameter sensitivity", "📄 Responses & data"]
)

with overview_tab:
    left, right = st.columns(2)

    with left:
        st.subheader("Repeat consistency vs Temperature")

        if "exact_consistency_rate" in filtered.columns:
            chart_data = (
                filtered.groupby(["temperature", "model"])[
                    "exact_consistency_rate"
                ]
                .mean()
                .unstack("model")
                .sort_index()
            )
            st.line_chart(chart_data)
        else:
            st.info("Consistency metric is unavailable.")

    with right:
        st.subheader("Repeat text similarity vs Temperature")

        if "mean_pairwise_text_similarity" in filtered.columns:
            chart_data = (
                filtered.groupby(["temperature", "model"])[
                    "mean_pairwise_text_similarity"
                ]
                .mean()
                .unstack("model")
                .sort_index()
            )
            st.line_chart(chart_data)
        else:
            st.info("Pairwise text similarity is unavailable.")

    st.subheader("Repeat consistency by sampling configuration")
    p1, p2 = st.columns(2)

    with p1:
        st.markdown("**Consistency by Top-K**")
        if "exact_consistency_rate" in filtered.columns:
            chart_data = (
                filtered.groupby(["top_k", "model"])[
                    "exact_consistency_rate"
                ]
                .mean()
                .unstack("model")
                .sort_index()
            )
            st.line_chart(chart_data)

    with p2:
        st.markdown("**Consistency by Top-P**")
        if "exact_consistency_rate" in filtered.columns:
            chart_data = (
                filtered.groupby(["top_p", "model"])[
                    "exact_consistency_rate"
                ]
                .mean()
                .unstack("model")
                .sort_index()
            )
            st.line_chart(chart_data)

    st.divider()
    st.subheader("Exact word-count compliance")
    if not responses.empty and "meets_word_count" in responses.columns:
        word_summary = (
            responses.groupby("model")
            .agg(
                responses=("meets_word_count", "size"),
                mean_word_count=("word_count", "mean"),
                exact_word_count_rate=("meets_word_count", "mean"),
            )
            .reset_index()
        )
        word_summary["exact_word_count_rate"] = (
            word_summary["exact_word_count_rate"] * 100
        )
        word_summary = word_summary.rename(
            columns={
                "model": "Model",
                "responses": "Successful responses",
                "mean_word_count": "Mean word count",
                "exact_word_count_rate": "Exactly target word count (%)",
            }
        )
        word_summary["Mean word count"] = word_summary["Mean word count"].round(2)
        word_summary["Exactly target word count (%)"] = (
            word_summary["Exactly target word count (%)"].round(2)
        )
        st.dataframe(word_summary, use_container_width=True, hide_index=True)
        chart_data = (
            responses.groupby("model")["meets_word_count"]
            .mean()
            .mul(100)
            .rename("Exact word-count compliance (%)")
        )
        st.bar_chart(chart_data)
        targets = sorted(responses["word_count_target"].dropna().unique().tolist())
        st.caption(
            f"Target word count: {', '.join(map(str, targets))}. "
            "Words are counted by splitting on whitespace; punctuation is not "
            "handled specially. Compliance is a task-constraint metric, not a "
            "measure of story quality."
        )
    else:
        st.info("Word-count compliance is unavailable because response data is missing.")

    st.caption(
        "These overview charts summarize repeated-run consistency. "
        "They do not directly measure creativity or factual accuracy."
    )

# --------------------------------------------------
# Text similarity helper
# --------------------------------------------------
def normalize_text(text):
    return " ".join(str(text).lower().split())


def calculate_sensitivity(
    data,
    parameter,
    fixed_conditions,
    model_name,
):
    """Compare one representative response per parameter value."""
    subset = data[data["model"] == model_name].copy()

    for column, value in fixed_conditions.items():
        subset = subset[
            subset[column].round(6) == round(float(value), 6)
        ]

    if subset.empty:
        return pd.DataFrame()

    # Use the first successful repeat for each parameter value.
    sort_columns = [parameter]
    if "repeat" in subset.columns:
        sort_columns.append("repeat")

    representatives = (
        subset.sort_values(sort_columns)
        .drop_duplicates(subset=[parameter], keep="first")
        .sort_values(parameter)
    )

    rows = []
    previous_value = None
    previous_response = None

    for _, row in representatives.iterrows():
        value = row[parameter]
        response = normalize_text(row["response"])

        if not response:
            continue

        if previous_response is not None:
            similarity = SequenceMatcher(
                None, previous_response, response
            ).ratio()

            rows.append(
                {
                    "Model": model_name,
                    "Comparison": f"{previous_value:g} → {value:g}",
                    "From": previous_value,
                    "To": value,
                    "Similarity": similarity,
                    "Text difference": 1 - similarity,
                }
            )

        previous_value = value
        previous_response = response

    return pd.DataFrame(rows)


# --------------------------------------------------
# Parameter sensitivity
# --------------------------------------------------
with sensitivity_tab:
    st.subheader("Compare outputs across parameter values")
    st.write(
        "These charts compare the wording of responses between "
        "adjacent parameter values while holding the other sampling "
        "settings constant."
    )

    if responses.empty or "response" not in responses.columns:
        st.warning(
            "No successful response data is available. "
            "Check results/responses.csv."
        )
    else:
        response_models = sorted(
            responses["model"].dropna().unique().tolist()
        )

        available_temps = sorted(
            responses["temperature"].dropna().unique().tolist()
        )
        available_topk = sorted(
            responses["top_k"].dropna().unique().tolist()
        )
        available_topp = sorted(
            responses["top_p"].dropna().unique().tolist()
        )

        if not response_models:
            st.warning("No models are available for comparison.")
        else:
            st.markdown("### 1. Temperature sensitivity")
            st.caption(
                "Top-K and Top-P are fixed. The temperature values "
                "are compared in ascending order."
            )

            t1, t2 = st.columns(2)
            with t1:
                fixed_topk_temp = st.selectbox(
                    "Fixed Top-K",
                    available_topk,
                    index=(
                        available_topk.index(40)
                        if 40 in available_topk else 0
                    ),
                    key="fixed_topk_temp",
                )
            with t2:
                fixed_topp_temp = st.selectbox(
                    "Fixed Top-P",
                    available_topp,
                    index=(
                        available_topp.index(0.9)
                        if 0.9 in available_topp else 0
                    ),
                    key="fixed_topp_temp",
                )

            temp_results = []
            for model in response_models:
                result = calculate_sensitivity(
                    responses,
                    "temperature",
                    {
                        "top_k": fixed_topk_temp,
                        "top_p": fixed_topp_temp,
                    },
                    model,
                )
                if not result.empty:
                    temp_results.append(result)

            if temp_results:
                temp_df = pd.concat(temp_results, ignore_index=True)
                st.bar_chart(
                    temp_df.pivot(
                        index="Comparison",
                        columns="Model",
                        values="Similarity",
                    )
                )
                st.dataframe(temp_df, use_container_width=True)
            else:
                st.warning(
                    "No matching responses for these settings."
                )

            st.divider()
            st.markdown("### 2. Top-K sensitivity")
            st.caption(
                "Temperature and Top-P are fixed. Top-K values "
                "are compared in ascending order."
            )

            k1, k2 = st.columns(2)
            with k1:
                fixed_temp_topk = st.selectbox(
                    "Fixed Temperature",
                    available_temps,
                    index=(
                        available_temps.index(0.7)
                        if 0.7 in available_temps else 0
                    ),
                    key="fixed_temp_topk",
                )
            with k2:
                fixed_topp_topk = st.selectbox(
                    "Fixed Top-P",
                    available_topp,
                    index=(
                        available_topp.index(0.9)
                        if 0.9 in available_topp else 0
                    ),
                    key="fixed_topp_topk",
                )

            topk_results = []
            for model in response_models:
                result = calculate_sensitivity(
                    responses,
                    "top_k",
                    {
                        "temperature": fixed_temp_topk,
                        "top_p": fixed_topp_topk,
                    },
                    model,
                )
                if not result.empty:
                    topk_results.append(result)

            if topk_results:
                topk_df = pd.concat(topk_results, ignore_index=True)
                st.bar_chart(
                    topk_df.pivot(
                        index="Comparison",
                        columns="Model",
                        values="Similarity",
                    )
                )
                st.dataframe(topk_df, use_container_width=True)
            else:
                st.warning(
                    "No matching responses for these settings."
                )

            st.divider()
            st.markdown("### 3. Top-P sensitivity")
            st.caption(
                "Temperature and Top-K are fixed. Top-P values "
                "are compared in ascending order."
            )

            pp1, pp2 = st.columns(2)
            with pp1:
                fixed_temp_topp = st.selectbox(
                    "Fixed Temperature",
                    available_temps,
                    index=(
                        available_temps.index(0.7)
                        if 0.7 in available_temps else 0
                    ),
                    key="fixed_temp_topp",
                )
            with pp2:
                fixed_topk_topp = st.selectbox(
                    "Fixed Top-K",
                    available_topk,
                    index=(
                        available_topk.index(40)
                        if 40 in available_topk else 0
                    ),
                    key="fixed_topk_topp",
                )

            topp_results = []
            for model in response_models:
                result = calculate_sensitivity(
                    responses,
                    "top_p",
                    {
                        "temperature": fixed_temp_topp,
                        "top_k": fixed_topk_topp,
                    },
                    model,
                )
                if not result.empty:
                    topp_results.append(result)

            if topp_results:
                topp_df = pd.concat(topp_results, ignore_index=True)
                st.bar_chart(
                    topp_df.pivot(
                        index="Comparison",
                        columns="Model",
                        values="Similarity",
                    )
                )
                st.dataframe(topp_df, use_container_width=True)
            else:
                st.warning(
                    "No matching responses for these settings."
                )

            st.info(
                "Similarity ranges from 0 to 1. Higher values mean "
                "more similar wording; lower values mean more "
                "textual difference. These scores do not measure "
                "accuracy, creativity, or quality."
            )

# --------------------------------------------------
# Metrics table and generated responses
# --------------------------------------------------
with data_tab:
    st.subheader("Sampling configuration metrics")

    columns = [
        "model",
        "prompt_id",
        "temperature",
        "top_k",
        "top_p",
        "successful_repeats",
        "unique_normalized_responses",
        "exact_consistency_rate",
        "mean_pairwise_text_similarity",
        "json_valid_rate",
        "reference_lexical_similarity",
    ]
    available_columns = [
        column for column in columns if column in filtered.columns
    ]
    st.dataframe(
        filtered[available_columns],
        use_container_width=True,
    )

    if not responses.empty:
        st.subheader("Generated responses")

        response_models = sorted(
            responses["model"].dropna().unique().tolist()
        )
        selected_response_model = st.selectbox(
            "Choose a model",
            response_models,
            key="response_model",
        )

        shown = responses[
            responses["model"] == selected_response_model
        ].copy()

        response_columns = [
            "model",
            "prompt_id",
            "temperature",
            "top_k",
            "top_p",
            "repeat",
            "response",
            "word_count",
            "word_count_target",
            "meets_word_count",
            "error",
        ]
        response_columns = [
            column for column in response_columns
            if column in shown.columns
        ]
        st.dataframe(
            shown[response_columns],
            use_container_width=True,
        )

st.caption(
    "Experiment limitation: findings use one creative prompt and "
    "a limited number of repeats. Text similarity is not a "
    "hallucination detector or a factual accuracy score."
)

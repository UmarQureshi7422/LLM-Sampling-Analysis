
"""Transparent, lightweight metrics for repeated LLM outputs."""
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher


def is_json_like(text):
    """Return True only if the complete response parses as JSON."""
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def normalize(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def count_words(text):
    """Count words separated by whitespace."""
    return len((text or "").split())


def evaluate_run_group(records, prompts):
    """Aggregate repeat-level outputs by model, prompt and sampling configuration.

    Word-count compliance is measured against a prompt's word_count requirement.
    If no requirement is specified, the metric is left as None.

    Lexical similarity is not a factuality or hallucination detector.
    """
    prompt_map = {str(p["id"]): p for p in prompts}
    groups = defaultdict(list)

    for row in records:
        if row.get("error") is None and row.get("response") is not None:
            key = (
                row["model"], str(row["prompt_id"]), row["temperature"],
                row["top_k"], row["top_p"]
            )
            groups[key].append(row)

    output = []

    for key, rows in groups.items():
        model, prompt_id, temp, top_k, top_p = key
        answers = [r.get("response", "") for r in rows]
        normalized = [normalize(a) for a in answers]
        unique_count = len(set(normalized))

        pairwise = []
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                pairwise.append(
                    SequenceMatcher(None, normalized[i], normalized[j]).ratio()
                )

        prompt = prompt_map.get(prompt_id, {})
        reference = prompt.get("reference_answer")
        reference_similarity = None

        if reference:
            reference_similarity = sum(
                SequenceMatcher(None, normalize(a), normalize(reference)).ratio()
                for a in answers
            ) / len(answers)

        word_count_target = prompt.get("word_count")
        word_counts = [count_words(a) for a in answers]

        exact_word_count_rate = None
        if word_count_target is not None:
            exact_word_count_rate = sum(
                count == int(word_count_target) for count in word_counts
            ) / len(word_counts)

        output.append({
            "model": model,
            "prompt_id": prompt_id,
            "temperature": temp,
            "top_k": top_k,
            "top_p": top_p,
            "successful_repeats": len(rows),
            "unique_normalized_responses": unique_count,
            "exact_consistency_rate": (
                1.0 - ((unique_count - 1) / max(len(rows) - 1, 1))
                if len(rows) > 1 else 1.0
            ),
            "mean_pairwise_text_similarity": (
                sum(pairwise) / len(pairwise) if pairwise else 1.0
            ),
            "json_valid_rate": (
                sum(is_json_like(a) for a in answers) / len(answers)
            ),
            "mean_word_count": sum(word_counts) / len(word_counts),
            "exact_word_count_rate": exact_word_count_rate,
            # Alias for clarity in the dashboard and report.
            "word_count_compliance_rate": exact_word_count_rate,
            "reference_lexical_similarity": reference_similarity,
            "hallucination_note": (
                "Lexical similarity is only a rough screening signal; "
                "not a hallucination detector."
                if reference else
                "No reference answer supplied; hallucination requires "
                "external/task-specific verification."
            )
        })

    return output

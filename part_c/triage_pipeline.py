"""Offline ticket triage: privacy, retries, validation, budget accounting and routing."""
from collections import Counter
from pathlib import Path
import random

import pandas as pd
import mock_llm
from part_c.guardrails import (
    EMAIL_REGEX, PHONE_REGEX, ORDER_REGEX, check_input_guardrails, redact_pii, verify_zero_pii_remaining,
)
from part_c.classifier import (
    ALLOWED_CATEGORIES, BASE_DELAY, FEW_SHOT_PROMPT, call_with_retry, validate_response,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_COLUMNS = [
    "ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used",
]
HUMAN_STATUSES = {
    "skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted", "flagged_model_error",
}
CLASSIFIED_STATUSES = {"classified_json", "classified_keyword"}


def load_customer_segments(segments_path):
    """Load Part B's segment names and parse CSV booleans without string truthiness."""
    segments = pd.read_csv(segments_path, dtype=str, keep_default_na=False)
    expected = {"customer_id", "segment_name", "is_premium"}
    if not expected.issubset(segments.columns):
        raise ValueError(f"Segment file needs columns {sorted(expected)}")
    if segments["customer_id"].duplicated().any():
        raise ValueError("Segment file must contain one row per customer")
    result = {}
    for row in segments.to_dict("records"):
        premium_text = str(row["is_premium"]).strip().lower()
        if premium_text not in {"true", "false", "1", "0"}:
            raise ValueError("is_premium must be a CSV boolean or 0/1")
        result[row["customer_id"]] = {
            "segment_name": row["segment_name"], "is_premium": premium_text in {"true", "1"},
        }
    return result


def route_ticket(status, category, is_premium):
    """Human review takes precedence, followed by fraud, Premium and ordinary support."""
    if status in HUMAN_STATUSES:
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if is_premium:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


def summarise_run(results, total_tokens):
    """Compute accuracy only on tickets accepted by a rule, with a fixed confusion layout."""
    classified = results[results["status"].isin(CLASSIFIED_STATUSES)]
    count = len(classified)
    correct = int(classified["category"].eq(classified["true_category"]).sum())
    confusion = pd.crosstab(classified["true_category"], classified["category"]).reindex(
        index=ALLOWED_CATEGORIES, columns=ALLOWED_CATEGORIES, fill_value=0,
    )
    return {
        "rows": len(results), "classified": count, "correct": correct,
        "accuracy": correct / count if count else None,
        "total_tokens": int(total_tokens),
        "tokens_per_classified_ticket": total_tokens / count if count else None,
        "status_counts": {str(k): int(v) for k, v in results["status"].value_counts().items()},
        "routing_counts": {str(k): int(v) for k, v in results["route"].value_counts().items()},
        "priority_counts": {str(k): int(v) for k, v in results["priority"].value_counts().items()},
        "confusion_categories": ALLOWED_CATEGORIES,
        "confusion_matrix": confusion.to_numpy().tolist(),
    }


def run_ticket_triage(
    tickets_path=ROOT / "data" / "raw" / "tickets.csv",
    segments_path=ROOT / "outputs" / "customer_segments.csv",
    prompt=FEW_SHOT_PROMPT,
    token_budget=100000,
    retry_logger=None,
    random_seed=42,
):
    """Process file order; None disables the budget. Return (results, billed token total).

    The assignment's budget admits a request when usage SO FAR is below the budget.
    Its last response may cross the ceiling; successful-response usage is recorded,
    and every later eligible ticket is handed to a human without another call.
    Transient failures have no returned usage and hence no billable tokens in this mock.
    DataFrame.attrs['audit'] contains retry timing, redaction and privacy evidence.
    """
    if token_budget is not None and (not isinstance(token_budget, int) or token_budget < 0):
        raise ValueError("token_budget must be None or a non-negative integer")
    tickets = pd.read_csv(tickets_path, keep_default_na=False, dtype=str)
    segments = load_customer_segments(segments_path)
    verify_zero_pii_remaining(prompt)
    rng = random.Random(random_seed)
    running_tokens = 0
    results = []
    retry_events = []
    redaction_counts = Counter({"email": 0, "phone": 0, "order_id": 0, "total": 0})
    pii_left = Counter({"email": 0, "phone": 0, "order_id": 0})
    successful_calls = 0
    permanent_errors = 0
    # Immediately precedes the independent full run, restoring the mock failure schedule.
    mock_llm.reset()
    for row in tickets.to_dict("records"):
        tid, cid, raw_text = row["ticket_id"], row["customer_id"], row["text"]
        # Length is measured on raw text; PII is counted across ALL 164 tickets.
        input_status, _ = check_input_guardrails(raw_text)
        redacted_text, counts = redact_pii(raw_text)
        redaction_counts.update(counts)
        verify_zero_pii_remaining(redacted_text)
        pii_left.update({"email": len(EMAIL_REGEX.findall(redacted_text)),
                         "phone": len(PHONE_REGEX.findall(redacted_text)),
                         "order_id": len(ORDER_REGEX.findall(redacted_text))})
        category, tokens_used = "", 0
        status = input_status
        admitted = lambda: token_budget is None or running_tokens < token_budget
        if input_status == "eligible":
            if not admitted():
                status = "budget_exhausted"
            else:
                messages = [{"role": "system", "content": prompt},
                            {"role": "user", "content": redacted_text}]
                assert messages[1]["content"] == redacted_text
                assert len(messages) == 2 and messages[1]["role"] == "user"
                for message in messages:
                    verify_zero_pii_remaining(message["content"])
                try:
                    response = call_with_retry(
                        messages, tid, logger=retry_logger, rng=rng,
                        retry_events=retry_events, before_call=admitted,
                    )
                except RuntimeError:
                    # Preserves a complete ticket output if a provider stays unavailable.
                    status = "flagged_model_error"
                    permanent_errors += 1
                else:
                    if response is None:
                        status = "budget_exhausted"
                    else:
                        successful_calls += 1
                        tokens_used = int(response.usage.total_tokens)
                        running_tokens += tokens_used
                        status, predicted = validate_response(response.choices[0].message.content)
                        category = predicted or ""
        segment = segments.get(cid, {"segment_name": "", "is_premium": False})
        route, priority = route_ticket(status, category, segment["is_premium"])
        results.append({
            "ticket_id": tid, "customer_id": cid, "status": status, "category": category,
            "segment_name": segment["segment_name"], "priority": priority, "route": route,
            "tokens_used": tokens_used, "true_category": row.get("true_category", ""),
        })
    frame = pd.DataFrame(results, columns=OUTPUT_COLUMNS + ["true_category"])
    assert frame["ticket_id"].tolist() == tickets["ticket_id"].tolist()
    assert int(frame["tokens_used"].sum()) == running_tokens
    assert frame.loc[frame["status"].isin({"skipped_empty", "flagged_long", "budget_exhausted"}),
                     "tokens_used"].eq(0).all()
    for event in retry_events:
        delay = BASE_DELAY * (2 ** event["retry_number"])
        assert 0.9 * delay <= event["wait_seconds"] <= 1.15 * delay
    frame.attrs["audit"] = {
        "random_seed": random_seed, "token_budget": token_budget,
        "system_prompt_estimated_tokens": mock_llm.est_tokens(prompt),
        "successful_model_calls": successful_calls,
        "model_attempts": successful_calls + len(retry_events) + permanent_errors,
        "transient_errors_recovered": len(retry_events), "retry_events": retry_events,
        "permanent_model_errors": permanent_errors,
        "redaction_counts": dict(redaction_counts), "pii_remaining": dict(pii_left),
        "all_prompt_pii_checks_passed": True,
        "budget_policy": "Admit only if usage so far is below budget; last response may cross it.",
        "budget_overshoot_tokens": max(0, running_tokens - token_budget) if token_budget is not None else 0,
    }
    return frame, running_tokens

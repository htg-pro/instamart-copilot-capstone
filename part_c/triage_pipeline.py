"""part_c/triage_pipeline.py - Orchestrator for support ticket triage and routing."""
import os
import pandas as pd
import mock_llm
from part_c.guardrails import check_input_guardrails, redact_pii, verify_zero_pii_remaining
from part_c.classifier import (
    ZERO_SHOT_PROMPT, FEW_SHOT_PROMPT, call_with_retry, validate_response
)

def run_ticket_triage(
    tickets_path=os.path.join("data", "raw", "tickets.csv"),
    segments_path=os.path.join("outputs", "customer_segments.csv"),
    prompt=FEW_SHOT_PROMPT,
    token_budget=100000,
    retry_logger=None
):
    tickets_df = pd.read_csv(tickets_path, keep_default_na=False)

    # Load segments
    customer_segments = {}
    if os.path.exists(segments_path):
        seg_df = pd.read_csv(segments_path)
        for _, r in seg_df.iterrows():
            customer_segments[r["customer_id"]] = {
                "segment_name": r["segment_name"],
                "is_premium": bool(r["is_premium"])
            }

    mock_llm.reset()
    running_tokens = 0
    results = []

    for _, row in tickets_df.iterrows():
        tid = row["ticket_id"]
        cid = row["customer_id"]
        txt = row["text"]
        true_cat = row.get("true_category", "")

        # 1. Input Guardrails
        guard_status, est_tok = check_input_guardrails(txt)
        if guard_status in ["skipped_empty", "flagged_long"]:
            results.append({
                "ticket_id": tid,
                "customer_id": cid,
                "status": guard_status,
                "category": "",
                "true_category": true_cat,
                "tokens_used": 0
            })
            continue

        # 2. Token Budget Guardrail
        if running_tokens >= token_budget:
            results.append({
                "ticket_id": tid,
                "customer_id": cid,
                "status": "budget_exhausted",
                "category": "",
                "true_category": true_cat,
                "tokens_used": 0
            })
            continue

        # 3. PII Redaction
        redacted_txt, _ = redact_pii(txt)
        verify_zero_pii_remaining(redacted_txt)

        # 4. Model Call with Retry
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": redacted_txt}
        ]
        resp = call_with_retry(messages, tid, logger=retry_logger)
        call_tokens = resp.usage.total_tokens
        running_tokens += call_tokens

        # 5. Output Validation
        val_status, pred_category = validate_response(resp.choices[0].message.content)

        results.append({
            "ticket_id": tid,
            "customer_id": cid,
            "status": val_status,
            "category": pred_category if pred_category else "",
            "true_category": true_cat,
            "tokens_used": call_tokens
        })

    # 6. Routing and Priority Assignment
    out_rows = []
    for r in results:
        cid = r["customer_id"]
        status = r["status"]
        cat = r["category"]

        seg_info = customer_segments.get(cid, {"segment_name": "", "is_premium": False})
        seg_name = seg_info["segment_name"]
        is_prem = seg_info["is_premium"]

        if status in ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]:
            route = "human_review"
            priority = "NORMAL"
        elif cat == "fraud":
            route = "security_team"
            priority = "HIGH"
        elif is_prem:
            route = "premium_desk"
            priority = "HIGH"
        else:
            route = "support_queue"
            priority = "NORMAL"

        out_rows.append({
            "ticket_id": r["ticket_id"],
            "customer_id": cid,
            "status": status,
            "category": cat,
            "segment_name": seg_name if cat else (seg_name if status not in ["skipped_empty", "flagged_long"] else ""),
            "priority": priority,
            "route": route,
            "tokens_used": r["tokens_used"],
            "true_category": r["true_category"]
        })

    return pd.DataFrame(out_rows), running_tokens

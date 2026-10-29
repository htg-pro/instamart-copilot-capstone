"""part_c/run_part_c.py - End-to-end execution of Part C."""
import os
import sys
import pandas as pd

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mock_llm
from part_c.guardrails import check_input_guardrails, redact_pii, verify_zero_pii_remaining
from part_c.classifier import ZERO_SHOT_PROMPT, FEW_SHOT_PROMPT
from part_c.triage_pipeline import run_ticket_triage

def main():
    print("==================================================")
    print("     PART C: SUPPORT TICKET TRIAGE & ROUTING      ")
    print("==================================================")

    tickets_path = os.path.join("data", "raw", "tickets.csv")
    tickets_df = pd.read_csv(tickets_path, keep_default_na=False)
    print(f"Total raw tickets loaded: {len(tickets_df)}")

    # 1. Input Guardrails Audit
    print("\n--- Step 1: Input Guardrails Audit ---")
    empty_tickets = []
    long_tickets = []
    for _, row in tickets_df.iterrows():
        status, tok = check_input_guardrails(row["text"])
        if status == "skipped_empty":
            empty_tickets.append((row["ticket_id"], tok))
        elif status == "flagged_long":
            long_tickets.append((row["ticket_id"], tok))

    print(f"Empty tickets (skipped_empty): {len(empty_tickets)}")
    print(f"Long tickets (flagged_long): {len(long_tickets)}")
    for tid, tok in long_tickets:
        print(f"  {tid}: estimated tokens = {tok}")

    # 2. PII Redaction Audit
    print("\n--- Step 2: PII Redaction Audit ---")
    total_subs = {"email": 0, "phone": 0, "order_id": 0, "total": 0}
    for _, row in tickets_df.iterrows():
        redacted, counts = redact_pii(row["text"])
        verify_zero_pii_remaining(redacted)
        total_subs["email"] += counts["email"]
        total_subs["phone"] += counts["phone"]
        total_subs["order_id"] += counts["order_id"]
        total_subs["total"] += counts["total"]

    print(f"Substitutions by type:")
    print(f"  Emails masked: {total_subs['email']}")
    print(f"  Phone numbers masked: {total_subs['phone']}")
    print(f"  Order IDs masked: {total_subs['order_id']}")
    print(f"  Total substitutions: {total_subs['total']}")
    print("Zero unmasked PII remaining in prompts: Verified (Assertion Passed)")

    # 3. Resilient Retries & Zero-Shot vs Few-Shot Comparison
    print("\n--- Step 3: Zero-Shot vs Few-Shot Pipeline Comparison ---")

    retry_logs = []
    def log_retry(line):
        retry_logs.append(line)

    # (a) Zero-shot run
    print("\nRunning Zero-Shot Triage...")
    df_zero, tokens_zero = run_ticket_triage(
        prompt=ZERO_SHOT_PROMPT,
        token_budget=1000000,
        retry_logger=log_retry
    )
    print(f"TransientLLMErrors caught & recovered: {len(retry_logs)}")

    # Status counts
    print("\nZero-Shot Status Counts:")
    print(df_zero["status"].value_counts().to_string())

    classified_zero = df_zero[df_zero["status"].isin(["classified_json", "classified_keyword"])]
    acc_zero = (classified_zero["category"] == classified_zero["true_category"]).mean()
    correct_zero = (classified_zero["category"] == classified_zero["true_category"]).sum()
    print(f"\nZero-Shot Classified Tickets: {len(classified_zero)} / 147")
    print(f"Zero-Shot Accuracy: {acc_zero*100:.2f}% ({correct_zero} / {len(classified_zero)})")
    print(f"Zero-Shot Total Tokens: {tokens_zero}")
    print(f"Zero-Shot Tokens per Classified Ticket: {tokens_zero / len(classified_zero):.2f}")

    print("\nZero-Shot Confusion Matrix (classified tickets):")
    ct_zero = pd.crosstab(classified_zero["true_category"], classified_zero["category"], margins=True)
    print(ct_zero)

    # (b) Few-shot run
    retry_logs.clear()
    print("\nRunning Few-Shot Triage...")
    df_few, tokens_few = run_ticket_triage(
        prompt=FEW_SHOT_PROMPT,
        token_budget=1000000,
        retry_logger=log_retry
    )
    classified_few = df_few[df_few["status"].isin(["classified_json", "classified_keyword"])]
    acc_few = (classified_few["category"] == classified_few["true_category"]).mean()
    correct_few = (classified_few["category"] == classified_few["true_category"]).sum()
    print(f"Few-Shot Classified Tickets: {len(classified_few)} / 147")
    print(f"Few-Shot Accuracy: {acc_few*100:.2f}% ({correct_few} / {len(classified_few)})")
    print(f"Few-Shot Total Tokens: {tokens_few}")
    print(f"Few-Shot Tokens per Classified Ticket: {tokens_few / len(classified_few):.2f}")

    # 4. Token Budget Demonstration
    print("\n--- Step 4: Token Budget Demonstration (TOKEN_BUDGET = 6000) ---")
    df_budget, tokens_budget = run_ticket_triage(
        prompt=ZERO_SHOT_PROMPT,
        token_budget=6000
    )
    budget_demo_path = os.path.join("outputs", "triage_budget_demo.csv")
    out_cols = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
    df_budget[out_cols].to_csv(budget_demo_path, index=False)
    exhausted_count = (df_budget["status"] == "budget_exhausted").sum()
    print(f"Tickets ending budget_exhausted: {exhausted_count}")
    print(f"Wrote {budget_demo_path}")

    # 5. Final Few-Shot Triage & Routing
    print("\n--- Step 5: Final Few-Shot Triage & Routing (Outputs Generation) ---")
    final_results_path = os.path.join("outputs", "triage_results.csv")
    df_few[out_cols].to_csv(final_results_path, index=False)
    print(f"Wrote {final_results_path} ({len(df_few)} rows)")

    print("\nRouting Breakdown:")
    print(df_few["route"].value_counts().to_string())

    print("\nPriority Breakdown:")
    print(df_few["priority"].value_counts().to_string())

    print("\nPart C completed successfully!")

if __name__ == "__main__":
    main()

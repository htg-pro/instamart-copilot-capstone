"""part_d/run_part_d.py - End-to-end execution of Part D."""
import os
import sys
import json
import re
import pandas as pd

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from part_d.rag import (
    chunk_policy_documents, PolicyRetriever, evaluate_policy_rag
)
from part_d.agent import run_agent, resume_agent

def main():
    print("==================================================")
    print("     PART D: POLICY RAG ASSISTANT & REFUND AGENT  ")
    print("==================================================")

    # 1. Document Chunking & Retrieval Evaluation
    print("\n--- Step 1: Chunking Policy Documents (40 words, 10 overlap) ---")
    chunks_10 = chunk_policy_documents(overlap=10)
    print(f"Total chunks generated: {len(chunks_10)}")
    by_file = {}
    for c in chunks_10:
        by_file[c["source"]] = by_file.get(c["source"], 0) + 1
    for f, count in sorted(by_file.items()):
        print(f"  {f}: {count} chunks")

    retriever_10 = PolicyRetriever(chunks_10)

    # Query check
    test_q = "refund policy late delivery"
    test_ret = retriever_10.retrieve(test_q, k=3)
    print(f"\nTop 3 chunks for '{test_q}':")
    for r in test_ret:
        print(f"  Source: {r['source']}, Score: {r['score']:.4f}")

    # 2. Evaluate with 10-word overlap
    print("\n--- Step 2: Evaluating RAG with 10-Word Overlap ---")
    eval_res_10 = evaluate_policy_rag(retriever_10)
    print(f"Answerable Document Retrieval Accuracy: {eval_res_10['doc_hits']} / {eval_res_10['total_answerable']}")
    print(f"Answerable Grounded Fact Accuracy: {eval_res_10['fact_hits']} / {eval_res_10['total_answerable']}")
    print(f"Unanswerable Fallback Accuracy: {eval_res_10['fallback_hits']} / {eval_res_10['total_unanswerable']}")

    # 3. Evaluate with zero overlap (40 words, 0 overlap)
    print("\n--- Step 3: Evaluating RAG with Zero Overlap ---")
    chunks_0 = chunk_policy_documents(overlap=0)
    retriever_0 = PolicyRetriever(chunks_0)
    eval_res_0 = evaluate_policy_rag(retriever_0)
    print(f"Zero-Overlap Doc Retrieval Accuracy: {eval_res_0['doc_hits']} / {eval_res_0['total_answerable']}")
    print(f"Zero-Overlap Grounded Fact Accuracy: {eval_res_0['fact_hits']} / {eval_res_0['total_answerable']}")
    print(f"Zero-Overlap Fallback Accuracy: {eval_res_0['fallback_hits']} / {eval_res_0['total_unanswerable']}")

    failing_q = [d for d in eval_res_0["details"] if not d["fact_hit"] and d["expected_doc"] != "NONE"]
    print("\nQuestions broken by Zero-Overlap Chunking (boundary split):")
    for item in failing_q:
        print(f"  Question: \"{item['question']}\"")
        print(f"    Expected Fact: \"{item['expected_fact']}\"")
        print(f"    Returned Answer: \"{item['answer']}\"\n")

    # 4. ReAct Refund Agent Scenarios
    print("\n--- Step 4: Executing ReAct Refund Agent on 7 Scenarios ---")
    scenarios_path = os.path.join("data", "raw", "agent_scenarios.csv")
    scenarios_df = pd.read_csv(scenarios_path)

    agent_runs_path = os.path.join("outputs", "agent_runs.jsonl")
    refund_ledger_path = os.path.join("outputs", "refund_ledger.jsonl")

    # Clear previous runs
    os.makedirs("outputs", exist_ok=True)
    if os.path.exists(agent_runs_path):
        os.remove(agent_runs_path)
    if os.path.exists(refund_ledger_path):
        os.remove(refund_ledger_path)

    scenario_summary = []

    for _, row in scenarios_df.iterrows():
        sid = row["scenario_id"]
        msg = row["message"]
        exp_outcome = row["expected_outcome"]
        exp_refund = float(row["expected_refund_inr"])

        run_output = run_agent(msg, sid)

        # Extract evaluated refund amount from calculate_refund step or final answer
        actual_refund = 0.0
        calc_step = next((s for s in run_output["steps"] if s["name"] == "calculate_refund"), None)
        if calc_step and "refund_inr" in calc_step.get("observation", {}):
            actual_refund = float(calc_step["observation"]["refund_inr"])
        else:
            m = re.search(r"INR\s+([\d.]+)", run_output["final_answer"])
            if m:
                actual_refund = float(m.group(1))

        # Append to agent_runs.jsonl
        with open(agent_runs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_output) + "\n")

        scenario_summary.append({
            "scenario_id": sid,
            "status": run_output["status"],
            "expected_outcome": exp_outcome,
            "refund_amount": f"INR {actual_refund:.2f}",
            "expected_refund": f"INR {exp_refund:.2f}",
            "llm_calls": run_output["llm_calls"],
            "tokens_used": run_output["tokens_used"],
            "final_answer": run_output["final_answer"]
        })

    summary_df = pd.DataFrame(scenario_summary)
    print("\nScenario Results Table:")
    print(summary_df[["scenario_id", "status", "expected_outcome", "refund_amount", "expected_refund", "llm_calls", "tokens_used"]].to_string(index=False))

    # 5. Stop-Safely Demonstrations
    print("\n--- Step 5: Stop Safely Guardrail Demonstrations ---")
    s1_row = scenarios_df.iloc[0]

    # (a) Token limit demonstration
    res_tok = run_agent(s1_row["message"], s1_row["scenario_id"], run_id="demo_token_limit", token_limit=200)
    print(f"Token Limit Guardrail Demo (token_limit=200): Status = {res_tok['status']}")

    # (b) Timeout demonstration
    res_time = run_agent(s1_row["message"], s1_row["scenario_id"], run_id="demo_timeout", time_limit_seconds=0)
    print(f"Timeout Guardrail Demo (time_limit_seconds=0): Status = {res_time['status']}")

    # 6. Resumable State & Crash Recovery Demonstration
    print("\n--- Step 6: Crash & Resume State Demonstration ---")
    crash_run_id = "s1_crash_resume_demo"
    print("Executing S1 with simulated crash after Tool Step 2 (search_policy)...")
    try:
        run_agent(
            s1_row["message"],
            s1_row["scenario_id"],
            run_id=crash_run_id,
            crash_after_tool_step=2
        )
    except RuntimeError as ex:
        print(f"Caught Simulated Crash: {ex}")

    print("Calling resume_agent to recover execution from checkpoint...")
    resumed_res = resume_agent(crash_run_id)
    print(f"Resumed Run Final Status: {resumed_res['status']}")
    print(f"Resumed Run Final Answer: {resumed_res['final_answer']}")
    print(f"Complete Tool Execution Sequence: {resumed_res['tools_executed']}")

    print("\nPart D completed successfully!")

if __name__ == "__main__":
    main()

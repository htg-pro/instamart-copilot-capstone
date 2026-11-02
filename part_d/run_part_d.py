"""Reproduce every Part D result and persist its text-only evidence artifacts."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import mock_llm
from part_d.rag import chunk_policy_documents, PolicyRetriever, evaluate_policy_rag
from part_d.agent import SimulatedCrash, load_checkpoint, run_agent, resume_agent
from part_d.tools import TOOLS_SPEC, TOOL_MAP, calculate_refund

OUTPUTS = ROOT / "outputs"
BATCH_RUN_IDS = [f"part_d_batch_{i.lower()}" for i in ("S1", "S2", "S3", "S4", "S5", "S6", "S7")]
DEMO_RUN_IDS = ["part_d_demo_token_limit", "part_d_demo_timeout", "part_d_demo_crash_resume"]


def _append_jsonline(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def _reset_demo_ledger():
    """Replace only this reproducible batch's actions; preserve unrelated ledger records."""
    path = OUTPUTS / "refund_ledger.jsonl"
    retained = []
    own_run_ids = set(BATCH_RUN_IDS + DEMO_RUN_IDS)
    if path.exists():
        retained = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        retained = [record for record in retained if record.get("run_id") not in own_run_ids]
    path.write_text("".join(json.dumps(record) + "\n" for record in retained), encoding="utf-8")


def refund_from_result(result):
    amounts = [step["observation"]["refund_inr"] for step in result["steps"]
               if "refund_inr" in step["observation"]]
    return float(amounts[-1]) if amounts else 0.0


def build_report():
    """Run overlapping/non-overlapping RAG, seven cases, safety stops, and recovery."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    mock_llm.reset()
    _reset_demo_ledger()
    chunks = chunk_policy_documents()
    chunk_counts = dict(sorted(Counter(chunk["source"] for chunk in chunks).items()))
    expected_counts = {"refund_policy.txt": 4, "delivery_policy.txt": 3, "escalation_policy.txt": 3,
                       "payment_policy.txt": 3, "privacy_policy.txt": 2}
    assert chunk_counts == expected_counts and len(chunks) == 15
    retriever = PolicyRetriever(chunks)
    reference_query = "refund policy late delivery"
    reference_results = retriever.retrieve(reference_query)
    assert [chunk["source"] for chunk in reference_results] == ["refund_policy.txt"] * 3
    overlap_10 = evaluate_policy_rag(retriever)
    overlap_0 = evaluate_policy_rag(PolicyRetriever(chunk_policy_documents(overlap=0)))
    assert (overlap_10["doc_hits"], overlap_10["fact_hits"], overlap_10["fallback_hits"]) == (12, 12, 2)
    broken = [detail for detail in overlap_0["details"]
              if detail["expected_doc"] != "NONE" and not detail["fact_hit"]]
    assert broken

    scenario_rows = pd.read_csv(ROOT / "data/raw/agent_scenarios.csv").to_dict("records")
    (OUTPUTS / "agent_runs.jsonl").write_text("", encoding="utf-8")
    summaries = []
    for row, run_id in zip(scenario_rows, BATCH_RUN_IDS):
        result = run_agent(row["message"], row["scenario_id"], run_id=run_id)
        _append_jsonline(OUTPUTS / "agent_runs.jsonl", result)
        expected_status = "FALLBACK_MAX_ITERATIONS" if row["scenario_id"] == "S7" else row["expected_outcome"]
        actual_refund = refund_from_result(result)
        assert result["status"] == expected_status
        assert abs(actual_refund - float(row["expected_refund_inr"])) <= 0.01
        summaries.append({"scenario_id": row["scenario_id"], "status": result["status"],
                          "refund_inr": actual_refund, "expected_refund_inr": float(row["expected_refund_inr"]),
                          "llm_calls": result["llm_calls"], "tokens_used": result["tokens_used"],
                          "tools_executed": result["tools_executed"], "final_answer": result["final_answer"]})
    assert summaries[0]["llm_calls"] == 5
    assert summaries[-1]["llm_calls"] == 6 and summaries[-1]["tokens_used"] <= 6000

    s1 = scenario_rows[0]
    token_demo = run_agent(s1["message"], "S1", run_id=DEMO_RUN_IDS[0], token_limit=200)
    timeout_demo = run_agent(s1["message"], "S1", run_id=DEMO_RUN_IDS[1], time_limit_seconds=0)
    assert token_demo["status"] == "FALLBACK_TOKEN_LIMIT"
    assert timeout_demo["status"] == "FALLBACK_TIMEOUT" and timeout_demo["llm_calls"] == 0
    assert len(token_demo["steps"]) < token_demo["llm_calls"]

    crash_run_id = DEMO_RUN_IDS[2]
    try:
        run_agent(s1["message"], "S1", run_id=crash_run_id, crash_after_tool_step=2)
    except SimulatedCrash as error:
        crash_message = str(error)
    else:
        raise AssertionError("The simulated crash did not occur.")
    before_resume = load_checkpoint(crash_run_id)
    assert before_resume["tools_executed"] == ["get_order", "search_policy"]
    resumed = resume_agent(crash_run_id)
    expected_sequence = ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert resumed["tools_executed"] == expected_sequence
    assert resumed["status"] == "REFUND" and refund_from_result(resumed) == 128.25
    assert resumed["llm_calls"] == 5 and resumed["resumed_llm_calls"] == 3
    ledger_before_repeat = (OUTPUTS / "refund_ledger.jsonl").read_text(encoding="utf-8")
    assert resume_agent(crash_run_id) == resumed
    assert (OUTPUTS / "refund_ledger.jsonl").read_text(encoding="utf-8") == ledger_before_repeat
    crash_ledger = [json.loads(line) for line in ledger_before_repeat.splitlines()
                    if json.loads(line).get("run_id") == crash_run_id]
    assert len(crash_ledger) == 1 and crash_ledger[0]["tool"] == "issue_refund"

    (OUTPUTS / "agent_demos.jsonl").write_text("", encoding="utf-8")
    for result in (token_demo, timeout_demo, resumed):
        _append_jsonline(OUTPUTS / "agent_demos.jsonl", result)

    boundary_examples = [
        {"question_id": "Q1", "source": "refund_policy.txt", "boundary_word": 40,
         "explanation": "The condition more than 45 minutes is in words 0-39, while the full-refund result "
         "starts at word 40. The mock selects the incomplete result sentence; the 30-69 chunk "
         "keeps the condition and result together."},
        {"question_id": "Q8", "source": "payment_policy.txt", "boundary_word": 40,
         "explanation": "The failed-payment condition ends in words 0-39 and within 48 hours begins at word 40. "
         "The zero-overlap retrieved context omits the timing chunk; overlapping words 30-69 retain both."},
    ]
    react_comparison = (
        f"The measured ReAct workflow uses 5 model calls and {summaries[0]['tokens_used']} estimated tokens for S1, "
        f"3 calls/{summaries[4]['tokens_used']} tokens for the cancelled order S5, and "
        f"2 calls/{summaries[5]['tokens_used']} tokens for NOT_FOUND S6. Each new call resends accumulated history, "
        "so its token cost grows with the number and size of observations. A plan-and-execute design could request "
        "the full conditional step list once, then execute validated tools locally and replan only on an error; "
        "a normal request could use one planning call plus an optional final response call. Its exact token saving "
        "is unmeasured because the supplied mock implements ReAct, not a planner. ReAct adapts immediately to "
        "NOT_FOUND or a cancelled status and stops before money is issued. A static plan must branch on those "
        "observations or replan; blindly following the remaining steps is unsafe. For this fixed refund workflow "
        "I would choose a validated conditional plan executed by deterministic Python rules, with human escalation "
        "above INR 1000. It reduces repeated model calls while retaining action-time checks. ReAct is preferable "
        "when policy or observations need more open-ended reasoning."
    )
    assert len(react_comparison.split()) <= 250
    own_ids = set(BATCH_RUN_IDS + DEMO_RUN_IDS)
    ledger = [json.loads(line) for line in (OUTPUTS / "refund_ledger.jsonl").read_text().splitlines()]
    batch_ledger = [entry for entry in ledger if entry.get("run_id") in own_ids]
    assert len(batch_ledger) == 4
    report = {
        "part": "D", "offline": True, "chunk_size_words": 40, "overlap_words": 10,
        "chunk_total": len(chunks), "chunk_counts": chunk_counts,
        "reference_query": reference_query, "reference_retrieval": reference_results,
        "rag_overlap_10": overlap_10, "rag_overlap_0": overlap_0,
        "zero_overlap_chunk_total": len(chunk_policy_documents(overlap=0)),
        "zero_overlap_failures": broken, "boundary_examples": boundary_examples,
        "tools_spec": TOOLS_SPEC, "tool_map_names": list(TOOL_MAP),
        "refund_boundary_checks": [
            {"order_amount": 100, "delivery_minutes": minutes,
             "refund_inr": calculate_refund(100, minutes)["refund_inr"]}
            for minutes in (45, 45.01, None)],
        "scenarios": summaries, "scenario_count": len(summaries),
        "limits": {"max_iterations": 6, "token_limit": 6000, "time_limit_seconds": 10},
        "token_limit_demo": token_demo, "timeout_demo": timeout_demo,
        "crash_resume": {"run_id": crash_run_id, "crash_message": crash_message,
                         "checkpoint_after_crash": before_resume,
                         "tools_before_crash": before_resume["tools_executed"],
                         "tools_after_resume": resumed["tools_executed"][2:],
                         "complete_tool_sequence": resumed["tools_executed"],
                         "tool_execution_count": len(resumed["steps"]),
                         "result": resumed, "refund_inr": refund_from_result(resumed),
                         "ledger_action_count": len(crash_ledger), "repeat_resume_no_duplicate": True},
        "ledger_actions_this_batch": batch_ledger,
        "react_vs_plan_and_execute": react_comparison,
        "artifacts": ["outputs/part_d_report.json", "outputs/agent_runs.jsonl",
                      "outputs/agent_demos.jsonl", "outputs/refund_ledger.jsonl", "state/part_d_*.json"],
        "verification": {"chunk_reference": True, "retrieval_grounding_reference": True,
                         "zero_overlap_has_failure": True, "seven_expected_scenarios": True,
                         "six_call_s7_with_default_budget": True, "independent_stop_limits": True,
                         "crash_resume_exactly_four_tools": True, "completed_resume_is_idempotent": True},
    }
    (OUTPUTS / "part_d_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    report = build_report()
    print("Part D: offline policy RAG and refund agent")
    print("Chunk counts:", report["chunk_counts"], "total:", report["chunk_total"])
    for overlap, key in ((10, "rag_overlap_10"), (0, "rag_overlap_0")):
        evaluation = report[key]
        print(f"Overlap={overlap}: expected document {evaluation['doc_hits']}/12; "
              f"expected fact {evaluation['fact_hits']}/12; uncovered fallback {evaluation['fallback_hits']}/2")
    print(pd.DataFrame(report["scenarios"])[
        ["scenario_id", "status", "refund_inr", "llm_calls", "tokens_used"]].to_string(index=False))
    print("Token-limit demo:", report["token_limit_demo"]["status"])
    print("Zero-timeout demo:", report["timeout_demo"]["status"])
    print("Crash/resume:", " -> ".join(report["crash_resume"]["complete_tool_sequence"]))
    print("Saved outputs/part_d_report.json, seven batch logs, three demo logs, and checkpoints.")
    return report


if __name__ == "__main__":
    main()

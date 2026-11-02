"""Verify the capstone's actual files, reference results and executed notebooks."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import joblib
import numpy as np
import pandas as pd

from part_a.audit import audit_raw_data
from part_a.clean import clean_customers_data, clean_orders_data
from part_a.db import run_sql_queries
from part_a.report import validate_sql_results
from part_a.validate import validate_part_a
from part_b.lapse_modeling import (FEATURE_COLUMNS, balance_training_only,
                                  build_cohort_and_features, positive_class_metrics,
                                  split_lapse_data)
from part_c.classifier import validate_response
from part_c.guardrails import (check_input_guardrails, redact_pii,
                               verify_zero_pii_remaining)


def foundation_regressions():
    # A newer NULL must not inherit a stale duration from the pending row.
    products = pd.DataFrame([{"product_id": "P001", "price_inr": 100, "category": "Grocery"}])
    orders = pd.DataFrame([
        ["ORD-100001", "C0001", "P001", 1, "2025-04-01 12:00:00", "pending", 0, 100, 99],
        ["ORD-100001", "C0001", "P001", 1, "2025-04-01 12:05:00", "completed", 0, 100, None],
        ["ORD-100002", "C0001", "P001", 1, "2025-04-02 12:05:00", "completed", 0, 100, 20],
        ["ORD-100003", "C0001", "P001", 1, "2025-04-03 12:05:00", "cancelled", 0, 100, 20],
    ], columns=["order_id", "customer_id", "product_id", "quantity", "order_ts", "status",
                "discount_pct", "order_amount", "delivery_minutes"])
    clean = clean_orders_data(orders, products).set_index("order_id")
    assert clean.loc["ORD-100001", "delivery_minutes"] == 20
    assert pd.isna(clean.loc["ORD-100003", "delivery_minutes"])
    customers = pd.DataFrame([
        ["C0001", "pune", "2024-01-01", 37, "M"],
        ["C0002", "PUNE", "2024-01-01", 38, "F"],
        ["C0003", " Pune ", "2024-01-01", None, ""],
    ], columns=["customer_id", "city", "signup_date", "age", "gender"])
    assert clean_customers_data(customers).loc[2, "age"] == 37.5


def lapse_regressions(features):
    X_train, X_test, y_train, y_test = split_lapse_data(features)
    balanced_X, balanced_y = balance_training_only(X_train, y_train)
    pd.testing.assert_frame_equal(balanced_X.iloc[:len(X_train)], X_train)
    pd.testing.assert_series_equal(balanced_y.iloc[:len(y_train)], y_train)
    assert set(balanced_X.index).isdisjoint(X_test.index)
    assert balanced_y.value_counts().to_dict() == {0: 570, 1: 570}
    # Future data outside September must change neither features nor labels.
    orders = pd.read_csv("data/clean/orders.csv")
    lapsed_id = features.loc[features.lapsed.eq(1), "customer_id"].iloc[0]
    future = orders.loc[orders.customer_id.eq(lapsed_id) & orders.status.eq("completed")].iloc[0].copy()
    future["order_id"] = "ORD-999998"
    future["order_ts"] = "2025-10-15 12:00:00"
    orders = pd.concat([orders, future.to_frame().T], ignore_index=True)
    with tempfile.TemporaryDirectory(prefix="instamart_lapse_") as temporary:
        path = Path(temporary) / "orders.csv"
        orders.to_csv(path, index=False)
        changed = build_cohort_and_features(orders_path=path)
    pd.testing.assert_frame_equal(features[["customer_id", *FEATURE_COLUMNS, "lapsed"]],
                                  changed[["customer_id", *FEATURE_COLUMNS, "lapsed"]])
    return X_test, y_test


def triage_regressions():
    assert validate_response('{"category":"shipping","reason":"billing"}') == ("flagged_invalid_output", None)
    assert validate_response('The issue is billing-related.') == ("classified_keyword", "billing")
    assert validate_response('billing or fraud') == ("flagged_invalid_output", None)
    assert check_input_guardrails("   ")[0] == "skipped_empty"
    redacted, counts = redact_pii("Email user@example.com; call 9876543210 for ORD-100001.")
    assert counts == {"email": 1, "phone": 1, "order_id": 1, "total": 3}
    assert verify_zero_pii_remaining(redacted)


def main():
    required = ["generate_data.py", "mock_llm.py", "requirements.txt", ".gitignore", "README.md",
                "sql/queries.sql", "data/instamart.db", "outputs/customer_segments.csv",
                "outputs/lapse_model.joblib", "outputs/retention_list.csv", "outputs/triage_results.csv",
                "outputs/triage_budget_demo.csv", "outputs/agent_runs.jsonl", "outputs/refund_ledger.jsonl",
                "outputs/agent_demos.jsonl"]
    for path in required:
        assert Path(path).is_file(), f"Missing required file: {path}"
    assert Path("state").is_dir()
    expected_hashes = {
        "generate_data.py": "e5bd939dfa40402281d0285bdf755d3b75f42159265d78098e248e58683de6cd",
        "mock_llm.py": "77bf9bb7cb96e67f49d808cb2e84befb0e5a346833cd508df924437b606905b6",
    }
    for name, expected in expected_hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, f"Supplied file modified: {name}"
    _, audit, _ = audit_raw_data()
    assert audit == dict(raw_order_rows=13084, exact_duplicates=119, pending_rows=656,
                         distinct_order_ids=12309, non_numeric_amounts=120, negative_delivery_times=106,
                         blank_delivery_times=2541, blank_ages=61, ages_outside_range=6,
                         distinct_raw_cities=24, distinct_raw_genders=10, blank_genders=30)
    validate_part_a()
    assert validate_sql_results(run_sql_queries())
    foundation_regressions()

    reports = {part: json.loads(Path(f"outputs/part_{part}_report.json").read_text()) for part in "abcd"}
    features = build_cohort_and_features()
    assert len(features) == 1156 and features.lapsed.sum() == 443
    X_test, y_test = lapse_regressions(features)
    model = joblib.load("outputs/lapse_model.joblib")
    model_metrics = positive_class_metrics(y_test, model.predict(X_test))
    assert model_metrics["f1"] >= .55
    b = reports["b"]
    assert b["chosen_k"] in [3, 4, 5]
    refs = {2:(1820.8,.465),3:(1268.2,.445),4:(743.1,.526),5:(617.4,.481),
            6:(503.2,.482),7:(408.5,.416),8:(366.5,.412)}
    for row in b["kmeans_evaluation"]:
        inertia, silhouette = refs[row["k"]]
        assert abs(row["inertia"] - inertia) <= inertia * .05
        assert abs(row["silhouette"] - silhouette) <= .05
    assert np.allclose(b["pca_explained_variance_ratio"], [.634, .218], atol=.02)
    segments = pd.read_csv("outputs/customer_segments.csv")
    assert len(segments) == 1175 and segments.customer_id.is_unique
    assert list(segments.columns) == ["customer_id","recency","frequency","monetary","segment_id","segment_name","is_premium"]
    highest = segments.groupby("segment_id").monetary.mean().idxmax()
    assert segments.is_premium.eq(segments.segment_id.eq(highest)).all()
    retention = pd.read_csv("outputs/retention_list.csv")
    assert list(retention.columns) == ["customer_id","lapse_probability","actually_lapsed"]
    assert len(retention) == min(10, b["lapse_experiment"]["premium_test_customers"])
    assert retention.lapse_probability.is_monotonic_decreasing
    assert set(retention.customer_id) <= set(features.loc[X_test.index, "customer_id"])

    triage_regressions()
    c = reports["c"]
    assert c["input_audit"]["pii_substitutions"] == {"email":55,"phone":91,"order_id":132,"total":278}
    for key in ["zero_shot", "few_shot", "final_few_shot"]:
        batch = c[key]
        assert batch["classified"] == 147
        assert batch["status_counts"] == {"classified_json":135,"classified_keyword":12,
                                           "flagged_invalid_output":9,"skipped_empty":4,"flagged_long":4}
        assert batch["audit"]["transient_errors_recovered"] == 19
        assert batch["audit"]["permanent_model_errors"] == 0
        assert batch["audit"]["all_prompt_pii_checks_passed"]
        for retry in batch["audit"]["retry_events"]:
            base = .05 * 2 ** retry["retry_number"]
            assert base * .9 <= retry["wait_seconds"] <= base * 1.15
    assert abs(c["zero_shot"]["accuracy"] - 126/147) < 1e-10
    assert c["few_shot"]["accuracy"] == 1
    assert c["budget_demo"]["status_counts"]["budget_exhausted"] > 0
    final = pd.read_csv("outputs/triage_results.csv", keep_default_na=False)
    assert len(final) == 164
    assert list(final.columns) == ["ticket_id","customer_id","status","category","segment_name","priority","route","tokens_used"]
    premium = segments.set_index("customer_id").is_premium.to_dict()
    for row in final.itertuples():
        if row.status not in ["classified_json", "classified_keyword"]:
            expected_route, expected_priority = "human_review", "NORMAL"
        elif row.category == "fraud":
            expected_route, expected_priority = "security_team", "HIGH"
        elif premium.get(row.customer_id, False):
            expected_route, expected_priority = "premium_desk", "HIGH"
        else:
            expected_route, expected_priority = "support_queue", "NORMAL"
        assert (row.route, row.priority) == (expected_route, expected_priority)

    runs = [json.loads(line) for line in Path("outputs/agent_runs.jsonl").read_text().splitlines()]
    assert len(runs) == 7 and [r["scenario_id"] for r in runs] == [f"S{i}" for i in range(1,8)]
    expected = ["REFUND","REFUND","ESCALATE","DENIED","DENIED","NOT_FOUND","FALLBACK_MAX_ITERATIONS"]
    assert [run["status"] for run in runs] == expected
    assert runs[0]["llm_calls"] == 5 and runs[-1]["llm_calls"] == 6
    assert runs[-1]["tokens_used"] <= 6000
    for run in runs:
        assert all(key in run for key in ["status","final_answer","steps","llm_calls","tokens_used"])
    ledger = [json.loads(line) for line in Path("outputs/refund_ledger.jsonl").read_text().splitlines()]
    scenarios = pd.read_csv("data/raw/agent_scenarios.csv")
    for run, scenario in zip(runs, scenarios.itertuples()):
        assert run["scenario_id"] == scenario.scenario_id
        refunds = [step["observation"]["refund_inr"] for step in run["steps"]
                   if "refund_inr" in step["observation"]]
        assert abs((refunds[-1] if refunds else 0) - scenario.expected_refund_inr) < .01
        checkpoint = json.loads(Path(f"state/{run['run_id']}.json").read_text())
        assert checkpoint["run_id"] == run["run_id"]
        assert checkpoint["scenario_id"] == run["scenario_id"]
        assert checkpoint["llm_calls"] == run["llm_calls"] and checkpoint["tokens_used"] == run["tokens_used"]
        assert checkpoint["step_number"] == len(run["steps"])
        assert checkpoint["tools_executed"] == [step["name"] for step in run["steps"]]
        assert checkpoint["messages"] and checkpoint["result"]["status"] == run["status"]
        if run["status"] in ["REFUND", "ESCALATE"]:
            action = "issue_refund" if run["status"] == "REFUND" else "escalate_to_human"
            actions = [record for record in ledger if record.get("run_id") == run["run_id"]]
            assert len(actions) == 1 and actions[0]["tool"] == action
            assert abs(actions[0]["refund_inr"] - scenario.expected_refund_inr) < .01

    from part_d.rag import chunk_policy_documents, PolicyRetriever, evaluate_policy_rag
    from part_d.agent import run_agent, resume_agent
    from part_d.tools import execute_tool, calculate_refund
    d = reports["d"]
    chunks = chunk_policy_documents()
    assert len(chunks) == 15 and d["chunk_counts"] == {
        "refund_policy.txt":4,"delivery_policy.txt":3,"escalation_policy.txt":3,
        "payment_policy.txt":3,"privacy_policy.txt":2}
    retriever = PolicyRetriever(chunks)
    assert [r["source"] for r in retriever.retrieve("refund policy late delivery")] == ["refund_policy.txt"] * 3
    actual_rag = evaluate_policy_rag(retriever)
    zero_rag = evaluate_policy_rag(PolicyRetriever(chunk_policy_documents(overlap=0)))
    assert (actual_rag["doc_hits"],actual_rag["fact_hits"],actual_rag["fallback_hits"]) == (12,12,2)
    assert (zero_rag["doc_hits"],zero_rag["fact_hits"],zero_rag["fallback_hits"]) == (12,10,2)
    assert d["zero_overlap_failures"] and len(chunk_policy_documents(overlap=0)) == 13
    for reported, actual in [(d["rag_overlap_10"],actual_rag),(d["rag_overlap_0"],zero_rag)]:
        assert all(reported[key] == actual[key] for key in ["doc_hits","fact_hits","fallback_hits"])
    assert d["token_limit_demo"]["status"] == "FALLBACK_TOKEN_LIMIT"
    assert len(d["token_limit_demo"]["steps"]) < d["token_limit_demo"]["llm_calls"]
    assert d["timeout_demo"]["status"] == "FALLBACK_TIMEOUT"
    assert d["timeout_demo"]["llm_calls"] == 0 and not d["timeout_demo"]["steps"]
    crash = d["crash_resume"]
    sequence = ["get_order","search_policy","calculate_refund","issue_refund"]
    assert crash["tools_before_crash"] == sequence[:2] and crash["tools_after_resume"] == sequence[2:]
    assert crash["complete_tool_sequence"] == sequence and crash["tool_execution_count"] == 4
    assert crash["checkpoint_after_crash"]["tools_executed"] == sequence[:2]
    assert crash["result"]["status"] == "REFUND" and crash["refund_inr"] == 128.25
    assert crash["result"]["llm_calls"] == 5 and crash["result"]["resumed_llm_calls"] == 3
    ledger_before = Path("outputs/refund_ledger.jsonl").read_bytes()
    assert resume_agent(crash["run_id"]) == crash["result"]
    assert Path("outputs/refund_ledger.jsonl").read_bytes() == ledger_before
    demos = [json.loads(line) for line in Path("outputs/agent_demos.jsonl").read_text().splitlines()]
    assert [demo["status"] for demo in demos] == ["FALLBACK_TOKEN_LIMIT","FALLBACK_TIMEOUT","REFUND"]
    with tempfile.TemporaryDirectory(prefix="instamart_agent_") as temporary:
        timeout = run_agent(scenarios.iloc[0].message,"S1",time_limit_seconds=0,state_dir=temporary)
        token = run_agent(scenarios.iloc[0].message,"S1",token_limit=0,state_dir=temporary)
        iteration = run_agent(scenarios.iloc[0].message,"S1",max_iterations=0,state_dir=temporary)
        assert timeout["status"] == "FALLBACK_TIMEOUT" and timeout["llm_calls"] == 0
        assert token["status"] == "FALLBACK_TOKEN_LIMIT" and not token["steps"]
        assert iteration["status"] == "FALLBACK_MAX_ITERATIONS" and iteration["llm_calls"] == 0
    try:
        execute_tool("get_order",{"order_id":"ORD-100048","db_path":"/tmp/other.db"},"verify",1)
    except ValueError:
        pass
    else:
        raise AssertionError("Tool dispatch allowed an undeclared database override.")
    assert calculate_refund(100,45)["refund_inr"] == 0
    assert calculate_refund(100,45.01)["refund_inr"] == 100

    notebooks = []
    for part in "abcd":
        paths = list(Path(f"part_{part}").glob("*.ipynb"))
        assert len(paths) == 1, f"Expected one complete Part {part.upper()} notebook"
        path = paths[0]
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert len(code) >= 5 and all(cell["execution_count"] is not None for cell in code)
        assert not any(output["output_type"] == "error" for cell in code for output in cell["outputs"])
        assert not any(mime.startswith("image/") for cell in code for output in cell["outputs"] for mime in output.get("data", {}))
        notebooks.append({"path":str(path),"executed_code_cells":len(code)})
    headings = [line[3:] for line in Path("README.md").read_text().splitlines() if line.startswith("## ")]
    assert headings == ["Overview","Reproduce","Part A","Part B","Part C","Part D","Limitations and Next Steps"]
    result = {"status":"ALL PROJECT CHECKS PASSED","supplied_files_unmodified":True,
              "parts_passed":["A","B","C","D"],"selected_model_test_f1":model_metrics["f1"],
              "notebooks":notebooks,"regression_checks":["latest whole order row","fractional age median",
              "future label isolation","oversampling preserves original train rows","strict output validation",
              "PII masking","reference results","routing","agent limits and artifacts"]}
    Path("outputs/verification_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print("ALL PROJECT CHECKS PASSED")
    print(f"Selected model lapsed-class F1: {model_metrics['f1']:.4f}")
    print(f"Executed notebooks verified: {len(notebooks)}")


if __name__ == "__main__":
    main()

"""Five offline refund tools, JSON schemas, and a durable local action ledger."""
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3

from part_d.rag import PolicyRetriever, chunk_policy_documents

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "outputs/refund_ledger.jsonl"
DB_PATH = ROOT / "data/instamart.db"
_RETRIEVER = None
_ACTION_CONTEXT = ContextVar("refund_action_context", default=None)


def get_order(order_id, db_path=DB_PATH):
    """Read an order from the Part A database, returning its refund-relevant fields."""
    # Read-only mode avoids silently creating an empty database when Part A is absent.
    with sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": float(row[2]),
            "delivery_minutes": None if row[3] in (None, "") else float(row[3])}


def search_policy(query, retriever=None):
    """Retrieve three policy chunks with TF-IDF cosine similarity."""
    global _RETRIEVER
    if retriever is None:
        if _RETRIEVER is None:
            _RETRIEVER = PolicyRetriever(chunk_policy_documents())
        retriever = _RETRIEVER
    return {"results": retriever.retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    """Return the full order amount for delivery over 45 minutes, otherwise zero."""
    amount = float(order_amount)
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("Order amount must be a finite nonnegative number.")
    minutes = None if delivery_minutes is None else float(delivery_minutes)
    if minutes is not None and (not math.isfinite(minutes) or minutes < 0):
        raise ValueError("Delivery minutes must be a finite nonnegative number or null.")
    return {"refund_inr": amount if minutes is not None and minutes > 45 else 0.0}


def append_to_ledger(tool_name, order_id, refund_inr, ledger_path=LEDGER_PATH):
    """Append an action once per run/step key, durably, for safe local replay."""
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    context = _ACTION_CONTEXT.get()
    operation_id = context["operation_id"] if context else None
    if operation_id and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("operation_id") == operation_id:
                if (record["tool"], record["order_id"], record["refund_inr"]) != (
                        tool_name, order_id, float(refund_inr)):
                    raise ValueError("A run/step id cannot be reused for a different refund action.")
                return
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "tool": tool_name,
              "order_id": order_id, "refund_inr": float(refund_inr)}
    if context:
        record.update(context)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _validate_action(order_id, refund_inr, escalation):
    """Enforce refund rules at the action boundary even if a tool request is wrong."""
    order = get_order(order_id)
    if "error" in order:
        raise ValueError("Cannot refund an order that does not exist.")
    if order["status"] != "completed":
        raise ValueError("Only completed orders can be refunded.")
    eligible = calculate_refund(order["order_amount"], order["delivery_minutes"])["refund_inr"]
    amount = float(refund_inr)
    if not math.isfinite(amount) or amount <= 0 or abs(amount - eligible) > 1e-9:
        raise ValueError("Refund must equal the positive eligible order amount.")
    if escalation and amount <= 1000:
        raise ValueError("Only refunds above INR 1000 require escalation.")
    if not escalation and amount > 1000:
        raise ValueError("Refunds above INR 1000 must be escalated to a human.")
    return amount


def issue_refund(order_id, refund_inr, ledger_path=LEDGER_PATH):
    """Record an eligible completed-order refund up to INR 1000."""
    amount = _validate_action(order_id, refund_inr, escalation=False)
    append_to_ledger("issue_refund", order_id, amount, ledger_path)
    return {"refund_inr": amount}


def escalate_to_human(order_id, refund_inr, ledger_path=LEDGER_PATH):
    """Record an eligible refund above INR 1000 for a human lead without issuing it."""
    amount = _validate_action(order_id, refund_inr, escalation=True)
    append_to_ledger("escalate_to_human", order_id, amount, ledger_path)
    return {"refund_inr": amount}


TOOLS_SPEC = [
    {"name": "get_order", "description": "Read an order's status, amount and delivery minutes from SQLite.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"], "additionalProperties": False}},
    {"name": "search_policy", "description": "Search three policy chunks using TF-IDF cosine similarity.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                    "required": ["query"], "additionalProperties": False}},
    {"name": "calculate_refund", "description": "Calculate a full refund for delivery over 45 minutes or zero otherwise.",
     "parameters": {"type": "object", "properties": {
         "order_amount": {"type": "number", "minimum": 0},
         "delivery_minutes": {"type": ["number", "null"], "minimum": 0}},
         "required": ["order_amount", "delivery_minutes"], "additionalProperties": False}},
    {"name": "issue_refund", "description": "Record an eligible completed-order refund no greater than INR 1000.",
     "parameters": {"type": "object", "properties": {
         "order_id": {"type": "string"}, "refund_inr": {"type": "number", "exclusiveMinimum": 0, "maximum": 1000}},
         "required": ["order_id", "refund_inr"], "additionalProperties": False}},
    {"name": "escalate_to_human", "description": "Record an eligible refund above INR 1000 for human approval.",
     "parameters": {"type": "object", "properties": {
         "order_id": {"type": "string"}, "refund_inr": {"type": "number", "exclusiveMinimum": 1000}},
         "required": ["order_id", "refund_inr"], "additionalProperties": False}},
]

TOOL_MAP = {"get_order": get_order, "search_policy": search_policy,
            "calculate_refund": calculate_refund, "issue_refund": issue_refund,
            "escalate_to_human": escalate_to_human}


def validate_tool_arguments(name, arguments):
    """Enforce the published schema before dispatching model-supplied arguments."""
    specification = next((item for item in TOOLS_SPEC if item["name"] == name), None)
    if specification is None or name not in TOOL_MAP:
        raise ValueError("Unknown tool name.")
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object.")
    schema = specification["parameters"]
    if set(arguments) - set(schema["properties"]):
        raise ValueError("Tool arguments contain undeclared fields.")
    if set(schema["required"]) - set(arguments):
        raise ValueError("Tool arguments are missing required fields.")
    for field, value in arguments.items():
        rule = schema["properties"][field]
        kinds = rule["type"] if isinstance(rule["type"], list) else [rule["type"]]
        valid = (("null" in kinds and value is None)
                 or ("string" in kinds and isinstance(value, str))
                 or ("number" in kinds and isinstance(value, (int, float))
                     and not isinstance(value, bool) and math.isfinite(value)))
        if not valid:
            raise ValueError(f"Invalid type for tool argument {field}.")
        if value is None or not isinstance(value, (int, float)):
            continue
        if "minimum" in rule and value < rule["minimum"]:
            raise ValueError(f"Tool argument {field} is below its minimum.")
        if "maximum" in rule and value > rule["maximum"]:
            raise ValueError(f"Tool argument {field} is above its maximum.")
        if "exclusiveMinimum" in rule and value <= rule["exclusiveMinimum"]:
            raise ValueError(f"Tool argument {field} must exceed its minimum.")
    return True


def execute_tool(name, arguments, run_id, step_number):
    """Resolve a named tool and attach a stable side-effect id for this run's step."""
    validate_tool_arguments(name, arguments)
    context = {"run_id": run_id, "operation_id": f"{run_id}:step:{step_number}"}
    token = _ACTION_CONTEXT.set(context)
    try:
        return TOOL_MAP[name](**arguments)
    finally:
        _ACTION_CONTEXT.reset(token)

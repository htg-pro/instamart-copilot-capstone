"""part_d/tools.py - ReAct agent tool definitions, specifications, and ledger logging."""
import os
import sqlite3
import json
import time
from part_d.rag import PolicyRetriever, chunk_policy_documents

# Global retriever instance
_DEFAULT_CHUNKS = chunk_policy_documents()
_DEFAULT_RETRIEVER = PolicyRetriever(_DEFAULT_CHUNKS)

LEDGER_PATH = os.path.join("outputs", "refund_ledger.jsonl")
DB_PATH = os.path.join("data", "instamart.db")

def append_to_ledger(tool_name, order_id, refund_inr, ledger_path=LEDGER_PATH):
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    record = {
        "timestamp": time.time(),
        "tool": tool_name,
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def get_order(order_id, db_path=DB_PATH):
    """
    Lookup order details from the analytical database.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
        (order_id,)
    )
    row = cur.fetchone()
    conn.close()

    if row:
        deliv = None if row[3] is None or row[3] == "" else float(row[3])
        return {
            "order_id": row[0],
            "status": row[1],
            "order_amount": float(row[2]),
            "delivery_minutes": deliv
        }
    return {"error": "order not found"}

def search_policy(query, retriever=_DEFAULT_RETRIEVER):
    """
    Search policy documents using TF-IDF cosine similarity.
    """
    top_chunks = retriever.retrieve(query, k=3)
    return {"results": top_chunks}

def calculate_refund(order_amount, delivery_minutes):
    """
    Calculate eligible refund according to policy:
    100% refund if delivery > 45 minutes, 0 if <= 45 minutes.
    """
    if delivery_minutes is not None and float(delivery_minutes) > 45.0:
        return {"refund_inr": float(order_amount)}
    return {"refund_inr": 0.0}

def issue_refund(order_id, refund_inr, ledger_path=LEDGER_PATH):
    """
    Issue approved refund (<= INR 1000) and record in ledger.
    """
    append_to_ledger("issue_refund", order_id, refund_inr, ledger_path)
    return {"refund_inr": float(refund_inr)}

def escalate_to_human(order_id, refund_inr, ledger_path=LEDGER_PATH):
    """
    Escalate high-value refund (> INR 1000) to human lead and record in ledger.
    """
    append_to_ledger("escalate_to_human", order_id, refund_inr, ledger_path)
    return {"refund_inr": float(refund_inr)}

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Reads data/instamart.db and returns order status, order_amount and delivery_minutes.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Searches the policy documentation for relevant rules.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Computes refund eligibility and amount based on order amount and delivery minutes.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"}
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Issues automatic refund for amounts up to INR 1000 and appends to ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Escalates refund above INR 1000 to a human support lead and appends to ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

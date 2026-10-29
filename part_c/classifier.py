"""part_c/classifier.py - Prompts, resilient retry mechanism, and output validation."""
import json
import re
import random
import time
import mock_llm
from mock_llm import TransientLLMError

ALLOWED_CATEGORIES = ["billing", "delivery", "technical", "fraud"]
BASE_DELAY = 0.05

ZERO_SHOT_PROMPT = """CLASSIFY the support ticket into exactly one of: billing, delivery, technical, fraud.
You must respond strictly in valid JSON format:
{"category": "...", "reason": "..."}"""

FEW_SHOT_PROMPT = """CLASSIFY the support ticket into exactly one of: billing, delivery, technical, fraud.
You must respond strictly in valid JSON format:
{"category": "...", "reason": "..."}
EXAMPLES:
Ticket: I was charged twice for order [ORDER_ID]. Please refund the extra payment.
Response: {"category": "billing", "reason": "charged twice for order"}
Ticket: Someone hacked my account and made unauthorized transactions.
Response: {"category": "fraud", "reason": "unauthorized account takeover"}"""

def call_with_retry(messages, ticket_id, max_retries=3, logger=None):
    """
    Calls mock_llm.chat with exponential backoff and jitter.
    Wait time for retry n (n = 0, 1, 2):
        BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
    Logs one line per retry showing ticket id, retry number, and wait.
    """
    for retry_num in range(max_retries + 1):
        try:
            resp = mock_llm.chat(messages)
            return resp
        except TransientLLMError as e:
            if retry_num >= max_retries:
                raise RuntimeError(f"Exceeded max retries ({max_retries}) for ticket {ticket_id}") from e
            wait = BASE_DELAY * (2 ** retry_num) * random.uniform(0.9, 1.15)
            log_line = f"[RETRY] Ticket: {ticket_id} | Retry: {retry_num} | Wait: {wait:.4f}s | Reason: {str(e)}"
            if logger:
                logger(log_line)
            time.sleep(wait)

def validate_response(content):
    """
    Validates the model response:
    1. Parse JSON: if category in allowed categories -> status 'classified_json', category
    2. Fallback: extract words via re.findall(r'[a-z]+', content.lower()).
       If exactly one allowed category is present -> status 'classified_keyword', category
    3. Otherwise -> status 'flagged_invalid_output', category None
    """
    # 1. Try parsing JSON
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            cat = data.get("category")
            if cat in ALLOWED_CATEGORIES:
                return "classified_json", cat
    except Exception:
        pass

    # 2. Keyword fallback for non-JSON or conversational text
    words = re.findall(r"[a-z]+", (content or "").lower())
    matched_cats = [c for c in ALLOWED_CATEGORIES if c in words]
    if len(matched_cats) == 1:
        return "classified_keyword", matched_cats[0]

    # 3. Invalid output
    return "flagged_invalid_output", None

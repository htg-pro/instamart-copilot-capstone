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

def call_with_retry(messages, ticket_id, max_retries=3, logger=None,
                    rng=None, retry_events=None, before_call=None, sleeper=time.sleep):
    """
    Calls mock_llm.chat with exponential backoff and jitter.
    Wait time for retry n (n = 0, 1, 2):
        BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
    Logs one line per retry showing ticket id, retry number, and wait.
    """
    if not 0 <= max_retries <= 3:
        raise ValueError("A ticket may have at most three retries")
    rng = rng or random
    logger = logger or print
    for retry_num in range(max_retries + 1):
        # Admission is checked for every attempt, including retries.
        if before_call is not None and not before_call():
            return None
        try:
            resp = mock_llm.chat(messages)
            return resp
        except TransientLLMError as e:
            if retry_num >= max_retries:
                raise RuntimeError(f"Exceeded max retries ({max_retries}) for ticket {ticket_id}") from e
            wait = BASE_DELAY * (2 ** retry_num) * rng.uniform(0.9, 1.15)
            if retry_events is not None:
                retry_events.append({"ticket_id": ticket_id, "retry_number": retry_num,
                                     "wait_seconds": wait})
            logger(f"[RETRY] Ticket: {ticket_id} | Retry: {retry_num} | Wait: {wait:.6f}s")
            sleeper(wait)

def validate_response(content):
    """
    Validates the model response:
    1. Parse JSON: if category in allowed categories -> status 'classified_json', category
    2. Fallback: extract words via re.findall(r'[a-z]+', content.lower()).
       If exactly one allowed category is present -> status 'classified_keyword', category
    3. Otherwise -> status 'flagged_invalid_output', category None
    """
    # 1. Try parsing JSON
    if not isinstance(content, str):
        return "flagged_invalid_output", None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Only malformed JSON may use keyword recovery. Hyphenated words
        # such as "billing-related" must be separated with this regex.
        words = re.findall(r"[a-z]+", content.lower())
        matched_cats = [c for c in ALLOWED_CATEGORIES if c in words]
        if len(matched_cats) == 1:
            return "classified_keyword", matched_cats[0]
        return "flagged_invalid_output", None
    if isinstance(data, dict) and data.get("category") in ALLOWED_CATEGORIES:
        return "classified_json", data["category"]
    # Valid JSON with an invalid category or schema is never guessed from its reason.
    return "flagged_invalid_output", None

"""part_c/guardrails.py - Input guardrails and PII redaction for support tickets."""
import re
import mock_llm

EMAIL_REGEX = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_REGEX = re.compile(r"\b[6-9]\d{9}\b")
ORDER_REGEX = re.compile(r"ORD-\d{6}")

MAX_INPUT_TOKENS = 400

def check_input_guardrails(text):
    """
    Checks input guardrails before sending to the model:
    (a) Empty or only whitespace -> 'skipped_empty'
    (b) est_tokens(text) > 400 -> 'flagged_long'
    Otherwise returns 'eligible'.
    """
    if not (text or "").strip():
        return "skipped_empty", 0

    tokens = mock_llm.est_tokens(text)
    if tokens > MAX_INPUT_TOKENS:
        return "flagged_long", tokens

    return "eligible", tokens

def redact_pii(text):
    """
    Applies regex substitutions in exact order:
    1. email -> [EMAIL]
    2. phone -> [PHONE]
    3. order_id -> [ORDER_ID]
    Returns redacted_text and a dictionary of substitution counts.
    """
    counts = {}

    # 1. Email
    text_after_email, n_emails = EMAIL_REGEX.subn("[EMAIL]", text)
    counts["email"] = n_emails

    # 2. Phone
    text_after_phone, n_phones = PHONE_REGEX.subn("[PHONE]", text_after_email)
    counts["phone"] = n_phones

    # 3. Order ID
    redacted_text, n_orders = ORDER_REGEX.subn("[ORDER_ID]", text_after_phone)
    counts["order_id"] = n_orders
    counts["total"] = n_emails + n_phones + n_orders

    return redacted_text, counts

def verify_zero_pii_remaining(redacted_text):
    """
    Verifies that no unredacted email, phone, or order id remains in text.
    """
    emails = EMAIL_REGEX.findall(redacted_text)
    phones = PHONE_REGEX.findall(redacted_text)
    orders = ORDER_REGEX.findall(redacted_text)
    assert len(emails) == 0, f"Found leaked email in prompt: {emails}"
    assert len(phones) == 0, f"Found leaked phone in prompt: {phones}"
    assert len(orders) == 0, f"Found leaked order ID in prompt: {orders}"
    return True

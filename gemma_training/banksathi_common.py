"""Shared constants and helpers for the BankSathi Gemma-2B pipeline.

Kept in one place so the dataset generator, trainer and evaluator agree on the
exact prompt headers, JSON contracts and Gemma-2 chat wrapper used by
HybridAiService.kt.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata

VERIFIED_DATE = "2026-11-14"

# ---------------------------------------------------------------------------
# Authoritative source grounding matrix (guide page 4)
# ---------------------------------------------------------------------------
SOURCES = {
    "RBI_BC_BSBDA_01": (
        "Cash up to Rs 10,000 can be withdrawn via AePS thumb scan at a Bank Mitra "
        "(BC point). No extra BC fee or commission. Carry original Aadhaar and phone; "
        "printed receipt is mandatory."
    ),
    "RBI_BC_AEPS_04": (
        "Amounts above Rs 10,000 exceed the standard AePS BC limit. Route to the bank "
        "branch or ATM with passbook, Aadhaar and RuPay/ATM debit card."
    ),
    "RBI_BC_TRANSFER_03": (
        "Domestic cash deposit and transfer are supported at the Post Office (IPPB) or "
        "Bank Mitra. Carry Aadhaar, receiver account number and IFSC; take printed receipt."
    ),
    "RBI_BSBDA_DBT_02": (
        "PMJDY/BSBDA accounts need zero minimum balance, allow unlimited deposits and at "
        "least 4 free withdrawals a month. DBT credits can be checked at BC; passbook is "
        "printed at the branch."
    ),
    "RBI_KCC_MASTER_01": (
        "Kisan Credit Card crop and allied loans up to Rs 2,00,000 are collateral-free. "
        "Apply at the branch with Khatauni, 2 Aadhaar/PAN copies, passbook and 3 photos."
    ),
    "RBI_ZERO_LIABILITY_1930": (
        "OTP/PIN/CVV requests, AnyDesk or .apk installs, bit.ly KYC links, or AePS "
        "double-thumb without receipt are fraud. Call 1930 immediately and notify the "
        "bank within 3 working days for zero liability."
    ),
}


def source_snippet(source_id: str) -> str:
    return f"[{source_id} | Verified {VERIFIED_DATE}] {SOURCES[source_id]}"


# ---------------------------------------------------------------------------
# Prompt contracts (guide page 4) - must match HybridAiService.kt byte-for-byte
# ---------------------------------------------------------------------------
TASK_A_HEADER = (
    "You are BankSathi, an offline-first rural Indian banking assistant. Rephrase ONLY "
    "from Verified Context. If unsure, set abstained=true and task=\"NONE\".\n"
    "Core Rules: Cash <= 10000 -> BC_POINT; > 10000 -> BRANCH/ATM; KCC <= 2L -> BRANCH; "
    "Transfer -> IPPB/BC; OTP/KYC -> 1930.\n"
    "Return JSON ONLY with keys: replyEn, replyHi, task, suggestedAmountInr, "
    "isScamAlert, abstained."
)

TASK_B_HEADER = (
    "Analyze this banking SMS or call scenario for rural fraud in India. Advise calling "
    "1930 and reporting within 3 days on HIGH_RISK.\n"
    "Return JSON ONLY with keys: riskLevel, explanationEn, explanationHi, actionTipEn, "
    "actionTipHi."
)

NO_CONTEXT = "none (use Core Rules)"


def build_task_a_prompt(query: str, context_ids: list[str]) -> str:
    ctx = "\n".join(source_snippet(s) for s in context_ids) if context_ids else NO_CONTEXT
    return f"{TASK_A_HEADER}\nVerified Context:\n{ctx}\nUser: {query}"


def build_task_b_prompt(message: str) -> str:
    return f"{TASK_B_HEADER}\nMessage: {message}"


# ---------------------------------------------------------------------------
# Gemma-2 chat wrapper (guide page 4)
# ---------------------------------------------------------------------------
USER_TURN = "<start_of_turn>user\n"
MODEL_TURN = "<start_of_turn>model\n"
END_TURN = "<end_of_turn>"


def wrap_prompt(prompt: str) -> str:
    """Everything up to and including the template boundary (loss-masked)."""
    return f"{USER_TURN}{prompt}{END_TURN}\n{MODEL_TURN}"


def wrap_full(prompt: str, completion: str) -> str:
    return f"{wrap_prompt(prompt)}{completion}{END_TURN}"


def compact_json(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# JSON schemas
# ---------------------------------------------------------------------------
TASKS = ("WITHDRAW", "TRANSFER", "PASSBOOK_DBT", "LOAN_KCC", "NONE")
RISK_LEVELS = ("SAFE", "CAUTION", "HIGH_RISK")

TASK_A_KEYS = ("replyEn", "replyHi", "task", "suggestedAmountInr", "isScamAlert", "abstained")
TASK_B_KEYS = ("riskLevel", "explanationEn", "explanationHi", "actionTipEn", "actionTipHi")

REPLY_WORD_CAP = 45
EXPLANATION_WORD_CAP = 50


def word_count(text: str) -> int:
    return len(text.split())


DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
LATIN_RE = re.compile(r"[A-Za-z]")


def devanagari_ok(text: str, min_ratio: float = 0.85) -> bool:
    """True when text is predominantly Devanagari (allows digits, Rs symbol, a few acronyms)."""
    dev = len(DEVANAGARI_RE.findall(text))
    lat = len(LATIN_RE.findall(text))
    return dev > 0 and dev / (dev + lat) >= min_ratio


def validate_task_a(obj, strict_caps: bool = True) -> list[str]:
    errs = []
    if not isinstance(obj, dict):
        return ["not a JSON object"]
    if set(obj.keys()) != set(TASK_A_KEYS):
        errs.append(f"keys mismatch: {sorted(obj.keys())}")
        return errs
    if not isinstance(obj["replyEn"], str) or not obj["replyEn"].strip():
        errs.append("replyEn empty")
    if not isinstance(obj["replyHi"], str) or not obj["replyHi"].strip():
        errs.append("replyHi empty")
    if obj["task"] not in TASKS:
        errs.append(f"bad task {obj['task']!r}")
    amt = obj["suggestedAmountInr"]
    if isinstance(amt, bool) or not isinstance(amt, int) or amt < 0:
        errs.append("suggestedAmountInr must be non-negative int")
    for k in ("isScamAlert", "abstained"):
        if not isinstance(obj[k], bool):
            errs.append(f"{k} must be bool")
    if obj.get("abstained") is True and obj.get("task") != "NONE":
        errs.append("abstained=true requires task=NONE")
    if strict_caps and not errs:
        if word_count(obj["replyEn"]) > REPLY_WORD_CAP:
            errs.append("replyEn > 45 words")
        if word_count(obj["replyHi"]) > REPLY_WORD_CAP:
            errs.append("replyHi > 45 words")
    return errs


def validate_task_b(obj, strict_caps: bool = True) -> list[str]:
    errs = []
    if not isinstance(obj, dict):
        return ["not a JSON object"]
    if set(obj.keys()) != set(TASK_B_KEYS):
        return [f"keys mismatch: {sorted(obj.keys())}"]
    if obj["riskLevel"] not in RISK_LEVELS:
        errs.append(f"bad riskLevel {obj['riskLevel']!r}")
    for k in TASK_B_KEYS[1:]:
        if not isinstance(obj[k], str) or not obj[k].strip():
            errs.append(f"{k} empty")
    if strict_caps and not errs:
        for k in TASK_B_KEYS[1:]:
            if word_count(obj[k]) > EXPLANATION_WORD_CAP:
                errs.append(f"{k} > 50 words")
    return errs


# ---------------------------------------------------------------------------
# Dedupe / leakage hashing (guide page 3, rule 3)
# ---------------------------------------------------------------------------
def normalize_query(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith(("P", "S")))
    return " ".join(text.split())


def query_hash(text: str) -> str:
    return hashlib.sha256(normalize_query(text).encode("utf-8")).hexdigest()


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

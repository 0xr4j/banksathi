#!/usr/bin/env python3
"""BankSathi stratified dataset engine (guide pages 2-5).

Generates 1,000-3,000 Gemma-2 chat rows for two HybridAiService.kt contracts:

  * Task A  queryGemmaLocal()              -> replyEn/replyHi/task/suggestedAmountInr/isScamAlert/abstained
  * Task B  analyzeScamWithGemmaLocal()    -> riskLevel/explanationEn/explanationHi/actionTipEn/actionTipHi

Features
  * stratified quotas over 10 task + safety buckets (no class > 16%)
  * 3 speech registers: English (~38%), Devanagari Hindi (~24%), Hinglish + Whisper-tiny ASR noise (~38%)
  * threshold boundaries (Rs 9,999 / 10,000 / 10,001 and KCC Rs 2,00,000 / 2,50,000)
  * ~50% rows grounded with [SOURCE_ID | Verified DATE] CACHED_FAQ snippets, ~50% core-rule fallback
  * adversarial "never guess" abstention traps and 3-tier ScamGuard SMS (incl. SAFE hard negatives)
  * SHA-256 dedupe of normalised queries, stratified 85/15 split with zero train/eval leakage
  * every target validated: strict JSON schema, <=45-word replies, Devanagari replyHi

Usage:
  python3 dataset_generator.py --out-dir data --target-rows 1500
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

from banksathi_common import (
    SOURCES,
    build_task_a_prompt,
    build_task_b_prompt,
    compact_json,
    devanagari_ok,
    query_hash,
    validate_task_a,
    validate_task_b,
    wrap_full,
    write_jsonl,
)

# ---------------------------------------------------------------------------
# Stratified quotas (guide page 3). safe_abstention weight 0.16 (raise to 0.22 if
# eval safe_abstention_pct < 95%, see failure playbook on page 7).
# ---------------------------------------------------------------------------
BUCKET_WEIGHTS = {
    "loan_kcc": 0.14,
    "passbook_dbt_bsbda": 0.14,
    "transfer_ippb_bc": 0.14,
    "safe_abstention": 0.16,
    "withdraw_bc_le_10k": 0.12,
    "withdraw_branch_gt_10k": 0.10,
    "assistant_scam_alert": 0.10,
    "scam_high_risk": 0.04,
    "scam_caution": 0.03,
    "scam_safe": 0.03,
}

REGISTER_WEIGHTS = (("en", 0.38), ("hi", 0.24), ("hl", 0.38))
GROUNDED_RATE = 0.50
DISTRACTOR_RATE = 0.30
EVAL_FRACTION = 0.15
MAX_ATTEMPTS_PER_ROW = 60

# ---------------------------------------------------------------------------
# Slot vocabularies (EN, HI, Hinglish)
# ---------------------------------------------------------------------------
BANKS = [
    ("SBI", "एसबीआई"), ("PNB", "पीएनबी"), ("Bank of Baroda", "बैंक ऑफ़ बड़ौदा"),
    ("Bank of India", "बैंक ऑफ़ इंडिया"), ("Union Bank", "यूनियन बैंक"), ("Canara Bank", "केनरा बैंक"),
    ("Gramin Bank", "ग्रामीण बैंक"), ("Central Bank", "सेंट्रल बैंक"), ("Indian Bank", "इंडियन बैंक"),
    ("UCO Bank", "यूको बैंक"), ("Aryavart Bank", "आर्यावर्त बैंक"), ("Baroda UP Bank", "बड़ौदा यूपी बैंक"),
]
PLACES = [
    ("Sitapur", "सीतापुर"), ("Barabanki", "बाराबंकी"), ("Gonda", "गोंडा"), ("Bahraich", "बहराइच"),
    ("Hardoi", "हरदोई"), ("Unnao", "उन्नाव"), ("Basti", "बस्ती"), ("Ballia", "बलिया"), ("Jaunpur", "जौनपुर"),
    ("Azamgarh", "आज़मगढ़"), ("Deoria", "देवरिया"), ("Mirzapur", "मिर्ज़ापुर"), ("Banda", "बांदा"),
    ("Etah", "एटा"), ("Gaya", "गया"), ("Darbhanga", "दरभंगा"), ("Muzaffarpur", "मुज़फ़्फ़रपुर"),
    ("Vidisha", "विदिशा"), ("Sagar", "सागर"), ("Rewa", "रीवा"), ("Satna", "सतना"), ("Jhansi", "झाँसी"),
]
CITIES = [
    ("Delhi", "दिल्ली"), ("Mumbai", "मुंबई"), ("Surat", "सूरत"), ("Ludhiana", "लुधियाना"), ("Pune", "पुणे"),
    ("Kolkata", "कोलकाता"), ("Jaipur", "जयपुर"), ("Lucknow", "लखनऊ"), ("Bengaluru", "बेंगलुरु"),
    ("Ahmedabad", "अहमदाबाद"), ("Hyderabad", "हैदराबाद"), ("Chennai", "चेन्नई"),
]
RELATIONS = [  # (EN, HI possessive base, Hinglish)
    ("my mother", "मेरी माँ", "meri maa"), ("my father", "मेरे पिताजी", "mere papa"),
    ("my wife", "मेरी पत्नी", "meri biwi"), ("my son", "मेरे बेटे", "mere bete"),
    ("my daughter", "मेरी बेटी", "meri beti"), ("my grandmother", "मेरी दादी", "meri dadi"),
    ("my neighbour", "मेरे पड़ोसी", "mere padosi"), ("my brother", "मेरे भाई", "mere bhai"),
]
CROPS = [
    ("wheat", "गेहूँ", "gehun"), ("paddy", "धान", "dhaan"), ("mustard", "सरसों", "sarson"),
    ("sugarcane", "गन्ना", "ganna"), ("potato", "आलू", "aloo"), ("onion", "प्याज़", "pyaaz"),
    ("soybean", "सोयाबीन", "soyabean"), ("maize", "मक्का", "makka"), ("gram", "चना", "chana"),
]
SCHEMES = [
    ("PM-KISAN", "पीएम-किसान", "pm kisan"), ("MGNREGA wage", "मनरेगा मज़दूरी", "manrega"),
    ("LPG subsidy", "एलपीजी सब्सिडी", "gas subsidy"), ("old age pension", "वृद्धावस्था पेंशन", "budhapa pension"),
    ("scholarship", "छात्रवृत्ति", "scholarship"), ("widow pension", "विधवा पेंशन", "vidhwa pension"),
]
REMOTE_APPS = ["AnyDesk", "TeamViewer", "QuickSupport", "RustDesk"]
REMOTE_APPS_HI = {"AnyDesk": "एनीडेस्क", "TeamViewer": "टीमव्यूअर", "QuickSupport": "क्विकसपोर्ट", "RustDesk": "रस्टडेस्क"}

HL_NUMBER_WORDS = {
    1000: "ek hazaar", 2000: "do hazaar", 3000: "teen hazaar", 4000: "chaar hazaar",
    5000: "paanch hazaar", 6000: "chhe hazaar", 7000: "saat hazaar", 8000: "aath hazaar",
    10000: "das hazaar", 12000: "barah hazaar", 15000: "pandrah hazaar", 20000: "bees hazaar",
    25000: "pachchees hazaar", 30000: "tees hazaar", 50000: "pachaas hazaar", 75000: "pachhattar hazaar",
    100000: "ek lakh", 150000: "dedh lakh", 200000: "do lakh", 250000: "dhai lakh",
    300000: "teen lakh", 400000: "chaar lakh", 500000: "paanch lakh",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def inr(n: int) -> str:
    """Indian digit grouping: 250000 -> 2,50,000."""
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups + [tail])


def lakh_str(n: int) -> str | None:
    if n >= 100000 and n % 50000 == 0:
        v = n / 100000
        return f"{v:g}"
    return None


def amt_en(n: int, rng: random.Random) -> str:
    opts = [f"Rs {inr(n)}", f"Rs {n}", f"{n} rupees", f"₹{inr(n)}"]
    lk = lakh_str(n)
    if lk:
        opts += [f"Rs {lk} lakh", f"{lk} lakh rupees"]
    return rng.choice(opts)


def amt_hi(n: int, rng: random.Random) -> str:
    opts = [f"{n} रुपये", f"₹{inr(n)}", f"{inr(n)} रुपये"]
    lk = lakh_str(n)
    if lk:
        opts.append(f"{lk} लाख रुपये")
    return rng.choice(opts)


def amt_hl(n: int, rng: random.Random) -> str:
    opts = [f"{n} rupaye", f"{n} rs", f"{n}"]
    if n in HL_NUMBER_WORDS:
        opts += [f"{HL_NUMBER_WORDS[n]} rupaye", HL_NUMBER_WORDS[n]] * 2
    return rng.choice(opts)


def fmt_amt(reg: str, n: int, rng: random.Random) -> str:
    return {"en": amt_en, "hi": amt_hi, "hl": amt_hl}[reg](n, rng)


ASR_SUBS = [
    ("aadhaar", "aadhar"), ("passbook", "pasbuk"), ("rupaye", "rupees"), ("rupaye", "rupay"),
    ("account", "akount"), ("kisan", "kisaan"), ("hazaar", "hajar"), ("loan", "lone"),
    ("nikalne", "nikalane"), ("chahiye", "chaiye"), ("paisa", "paisaa"), ("bank mitra", "bankmitra"),
    ("kya", "kyaa"), ("hain", "he"),
]
ASR_FILLERS = ["haan ", "accha ", "suniye ", "bhaiya ", "madam ", "ji ", "arey "]


def asr_noise(text: str, rng: random.Random) -> str:
    """Simulate Whisper-tiny slips: lowercase, dropped '?', phonetic spellings, fillers."""
    if rng.random() < 0.25:  # some clean Hinglish too
        return text
    s = text.lower().replace("?", "").replace(",", "")
    for a, b in ASR_SUBS:
        if a in s and rng.random() < 0.45:
            s = s.replace(a, b, 1)
    if rng.random() < 0.3:
        s = rng.choice(ASR_FILLERS) + s
    if rng.random() < 0.15:  # word repetition stutter
        w = s.split()
        if len(w) > 2:
            i = rng.randrange(len(w))
            w.insert(i, w[i])
            s = " ".join(w)
    return " ".join(s.split())


def pick_register(rng: random.Random) -> str:
    r, acc = rng.random(), 0.0
    for reg, w in REGISTER_WEIGHTS:
        acc += w
        if r < acc:
            return reg
    return REGISTER_WEIGHTS[-1][0]


def cap_first(s: str) -> str:
    return s[:1].upper() + s[1:]


def acct_no(rng: random.Random) -> str:
    return str(rng.randint(10000, 99999999))


def last4(rng: random.Random) -> str:
    return f"{rng.randint(0, 9999):04d}"


def phone(rng: random.Random) -> str:
    # Masked (e.g. 67XXXXXX72) so the public dataset never names a real person's number as a scammer.
    # The same random draws are made as before, so every other row of the dataset stays identical.
    digits = str(rng.choice("6789")) + "".join(str(rng.randint(0, 9)) for _ in range(9))
    return digits[:2] + "XXXXXX" + digits[-2:]


def short_code(rng: random.Random, n: int = 6) -> str:
    alphabet = "abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(rng.choice(alphabet) for _ in range(n))


def sms_date(rng: random.Random) -> str:
    return f"{rng.randint(1, 28):02d}-{rng.randint(1, 12):02d}-26"


def task_a(reply_en, reply_hi, task, amount=0, scam=False, abstained=False) -> dict:
    return {
        "replyEn": reply_en,
        "replyHi": reply_hi,
        "task": task,
        "suggestedAmountInr": int(amount),
        "isScamAlert": scam,
        "abstained": abstained,
    }


def task_b(level, exp_en, exp_hi, tip_en, tip_hi) -> dict:
    return {
        "riskLevel": level,
        "explanationEn": exp_en,
        "explanationHi": exp_hi,
        "actionTipEn": tip_en,
        "actionTipHi": tip_hi,
    }


# ---------------------------------------------------------------------------
# WITHDRAW (shared query templates; target depends on the Rs 10,000 boundary)
# ---------------------------------------------------------------------------
WITHDRAW_Q = {
    "en": [
        "Can I withdraw {amt} at the BC point?",
        "I need {amt} cash from my account. Where should I go?",
        "How do I withdraw {amt} using my thumb at Bank Mitra?",
        "{Rel} needs {amt} in cash. Can the BC agent give it with Aadhaar?",
        "Is there any fee to withdraw {amt} at the AePS point in {place}?",
        "Want to take out {amt} from my Jan Dhan account near {place}.",
        "Can Bank Mitra give me {amt} today?",
        "Where can I withdraw {amt} in my village?",
        "I have to pay school fees of {amt}. How do I take the cash out of my {bank} account?",
        "Can I take {amt} from my {bank} account with Aadhaar thumb?",
    ],
    "hi": [
        "मुझे {amt} निकालने हैं, कहाँ जाऊँ?",
        "क्या बैंक मित्र से {amt} निकाल सकते हैं?",
        "अंगूठा लगाकर {amt} कैसे निकालें?",
        "{rel} को {amt} नकद चाहिए, बीसी पॉइंट पर मिलेगा?",
        "{place} में आधार से {amt} निकालने पर कोई फीस लगती है क्या?",
        "जन धन खाते से {amt} निकालने हैं।",
        "आज बैंक मित्र से {amt} मिल जाएँगे?",
        "{bank} खाते से {amt} नकद कैसे निकालूँ?",
        "बच्चों की फीस के लिए {amt} चाहिए, कहाँ से निकालूँ?",
    ],
    "hl": [
        "bhaiya {amt} nikalne hain",
        "{amt} nikalna hai bank mitra se ho jayega kya",
        "angootha laga ke {amt} nikal sakte hain kya",
        "{rel} ko {amt} chahiye aadhaar se nikalenge",
        "bc point pe {amt} nikalne ka kitna charge lagega",
        "jan dhan khate se {amt} nikalna hai {place} mein",
        "aaj bank mitra {amt} de dega kya",
        "{bank} khate se {amt} cash nikalna hai kahan jaun",
        "school fees ke liye {amt} nikalne hain",
    ],
}

WITHDRAW_BC_REPLIES = [
    (
        "You can withdraw Rs {a} at your nearest Bank Mitra (BC point) with Aadhaar thumb scan. "
        "There is no extra fee. Carry original Aadhaar and phone, and always take the printed receipt.",
        "आप नज़दीकी बैंक मित्र (बीसी पॉइंट) पर आधार अंगूठा लगाकर ₹{a} निकाल सकते हैं। कोई अतिरिक्त "
        "शुल्क नहीं लगता। असली आधार और फ़ोन साथ रखें और छपी रसीद ज़रूर लें।",
    ),
    (
        "Rs {a} is within the Rs 10,000 Bank Mitra limit, so the BC point is fine. Give your Aadhaar "
        "thumb scan, pay no extra fee, and collect the printed receipt.",
        "₹{a} बैंक मित्र की ₹10,000 सीमा के अंदर है, इसलिए बीसी पॉइंट ठीक है। आधार अंगूठा लगाएँ, "
        "कोई अतिरिक्त फीस न दें और छपी रसीद लें।",
    ),
]

WITHDRAW_BRANCH_REPLIES = [
    (
        "Rs {a} is above the Rs 10,000 Bank Mitra limit. Please withdraw it at your bank branch or ATM. "
        "Carry your passbook, Aadhaar and RuPay or ATM debit card.",
        "₹{a} बैंक मित्र की ₹10,000 सीमा से ज़्यादा है। कृपया यह राशि बैंक शाखा या एटीएम से निकालें। "
        "पासबुक, आधार और रुपे या एटीएम डेबिट कार्ड साथ रखें।",
    ),
    (
        "Bank Mitra can pay only up to Rs 10,000. For Rs {a}, go to your bank branch or use an ATM with "
        "your debit card. Take your passbook and Aadhaar along.",
        "बैंक मित्र केवल ₹10,000 तक ही दे सकते हैं। ₹{a} के लिए अपनी बैंक शाखा जाएँ या डेबिट कार्ड से "
        "एटीएम का उपयोग करें। पासबुक और आधार साथ ले जाएँ।",
    ),
]


def amount_bc(rng):
    r = rng.random()
    if r < 0.22:
        return rng.choice([9999, 10000, 9990, 9500, 9900])
    if r < 0.55:
        return rng.choice([500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 7500, 8000])
    if r < 0.9:
        return rng.randrange(1, 100) * 100
    return rng.randint(150, 9999)


def amount_branch(rng):
    r = rng.random()
    if r < 0.22:
        return rng.choice([10001, 10100, 10500, 11000, 10050])
    if r < 0.55:
        return rng.choice([12000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 75000, 100000])
    if r < 0.9:
        return rng.randrange(11, 200) * 1000
    return rng.randint(10002, 150000)


def fill(template: str, reg: str, rng: random.Random, **extra) -> str:
    """Fill common slots for the given register."""
    bank = rng.choice(BANKS)
    place = rng.choice(PLACES)
    rel = rng.choice(RELATIONS)
    city = rng.choice(CITIES)
    crop = rng.choice(CROPS)
    scheme = rng.choice(SCHEMES)
    idx = {"en": 0, "hi": 1, "hl": 2}[reg]
    slots = {
        "bank": bank[0] if reg != "hi" else bank[1],
        "place": place[0] if reg != "hi" else place[1],
        "city": city[0] if reg != "hi" else city[1],
        "rel": rel[idx],
        "Rel": cap_first(rel[0]),
        "crop": crop[idx],
        "scheme": scheme[idx],
        "app": rng.choice(REMOTE_APPS),
    }
    slots.update(extra)
    return template.format(**slots)


def finish_query(q: str, reg: str, rng: random.Random) -> str:
    return asr_noise(q, rng) if reg == "hl" else q


def gen_withdraw(rng, gt10k: bool):
    reg = pick_register(rng)
    n = amount_branch(rng) if gt10k else amount_bc(rng)
    q = fill(rng.choice(WITHDRAW_Q[reg]), reg, rng, amt=fmt_amt(reg, n, rng))
    en, hi = rng.choice(WITHDRAW_BRANCH_REPLIES if gt10k else WITHDRAW_BC_REPLIES)
    tgt = task_a(en.format(a=inr(n)), hi.format(a=inr(n)), "WITHDRAW", n)
    src = "RBI_BC_AEPS_04" if gt10k else "RBI_BC_BSBDA_01"
    return finish_query(q, reg, rng), reg, tgt, src


# ---------------------------------------------------------------------------
# TRANSFER (IPPB / BC)
# ---------------------------------------------------------------------------
TRANSFER_Q = {
    "en": [
        ("How can I send {amt} to my son in {city}?", "send"),
        ("I want to transfer {amt} to my brother's account. Where do I go?", "send"),
        ("Can I deposit {amt} cash into my account at the post office?", "deposit"),
        ("How do I send money to {rel}'s bank account from the village?", "send_noamt"),
        ("What do I need to transfer {amt} through Bank Mitra?", "send"),
        ("Can the IPPB post office send {amt} to {city}?", "send"),
        ("I want to deposit {amt} in {rel}'s account.", "deposit"),
        ("Can Bank Mitra in {place} deposit {amt} into my {bank} account?", "deposit"),
        ("Which documents are needed to send money to {city} from {place}?", "send_noamt"),
    ],
    "hi": [
        ("{city} में बेटे को {amt} कैसे भेजूँ?", "send"),
        ("भाई के खाते में {amt} भेजने हैं, कहाँ जाऊँ?", "send"),
        ("क्या डाकघर में अपने खाते में {amt} जमा कर सकते हैं?", "deposit"),
        ("गाँव से {rel} के बैंक खाते में पैसे कैसे भेजें?", "send_noamt"),
        ("बैंक मित्र से {amt} ट्रांसफर करने के लिए क्या चाहिए?", "send"),
        ("{rel} के खाते में {amt} जमा करने हैं।", "deposit"),
        ("{place} से {city} पैसे भेजने के लिए कौन से कागज़ चाहिए?", "send_noamt"),
    ],
    "hl": [
        ("bete ko {city} mein {amt} bhejna hai kaise bheju", "send"),
        ("bhai ke khate mein {amt} transfer karna hai", "send"),
        ("post office mein {amt} jama kar sakte hain kya", "deposit"),
        ("{rel} ke bank khate mein paisa kaise bheje", "send_noamt"),
        ("bank mitra se {amt} bhejne ke liye kya kya chahiye", "send"),
        ("ippb se {amt} {city} bhej sakte hain kya", "send"),
        ("{rel} ke khate mein {amt} jama karna hai", "deposit"),
        ("{place} se {city} paisa bhejna hai kya lagega", "send_noamt"),
    ],
}

TRANSFER_REPLIES = {
    "send": (
        "You can send Rs {a} from the Post Office (IPPB) or your Bank Mitra. Carry Aadhaar, the "
        "receiver's account number and IFSC code, and always take the printed receipt.",
        "आप डाकघर (आईपीपीबी) या बैंक मित्र से ₹{a} भेज सकते हैं। आधार, पाने वाले का खाता नंबर और "
        "आईएफ़एससी कोड साथ रखें, और छपी रसीद ज़रूर लें।",
    ),
    "send_noamt": (
        "You can send money from the Post Office (IPPB) or your Bank Mitra. Carry Aadhaar, the "
        "receiver's account number and IFSC code, and always take the printed receipt.",
        "आप डाकघर (आईपीपीबी) या बैंक मित्र से पैसे भेज सकते हैं। आधार, पाने वाले का खाता नंबर और "
        "आईएफ़एससी कोड साथ रखें, और छपी रसीद ज़रूर लें।",
    ),
    "deposit": (
        "You can deposit Rs {a} at the Post Office (IPPB) or your Bank Mitra. Carry Aadhaar and the "
        "account number, and take the printed receipt.",
        "आप डाकघर (आईपीपीबी) या बैंक मित्र पर ₹{a} जमा कर सकते हैं। आधार और खाता नंबर साथ रखें, "
        "और छपी रसीद लें।",
    ),
}


def amount_transfer(rng):
    r = rng.random()
    if r < 0.12:
        return rng.choice([9999, 10000, 10001])
    if r < 0.5:
        return rng.choice([500, 1000, 2000, 3000, 5000, 7000, 8000, 15000, 20000])
    return rng.randrange(2, 500) * 100


def gen_transfer(rng):
    reg = pick_register(rng)
    tmpl, kind = rng.choice(TRANSFER_Q[reg])
    n = amount_transfer(rng)
    q = fill(tmpl, reg, rng, amt=fmt_amt(reg, n, rng))
    en, hi = TRANSFER_REPLIES[kind]
    amount = 0 if kind == "send_noamt" else n
    tgt = task_a(en.format(a=inr(n)), hi.format(a=inr(n)), "TRANSFER", amount)
    return finish_query(q, reg, rng), reg, tgt, "RBI_BC_TRANSFER_03"


# ---------------------------------------------------------------------------
# PASSBOOK / DBT / BSBDA
# ---------------------------------------------------------------------------
PASSBOOK_Q = {
    "zero_bal": {
        "en": [
            "Do I need to keep minimum balance in my Jan Dhan account?",
            "Will the bank cut money if my {bank} account has zero balance?",
            "Is a BSBDA account free of minimum balance charges?",
            "My balance is only Rs {small}. Will {bank} charge a penalty?",
            "Can my Jan Dhan account in {place} stay at zero balance?",
        ],
        "hi": [
            "क्या जन धन खाते में न्यूनतम बैलेंस रखना ज़रूरी है?",
            "{bank} खाते में बैलेंस शून्य हो तो क्या पैसे कटेंगे?",
            "मेरे खाते में सिर्फ़ ₹{small} हैं, क्या जुर्माना लगेगा?",
            "{place} के जन धन खाते में ज़ीरो बैलेंस चलेगा?",
        ],
        "hl": [
            "jan dhan khate mein minimum balance rakhna padta hai kya",
            "{bank} khate mein zero balance ho to paisa katega kya",
            "khate mein sirf {small} rupaye hain penalty lagegi kya",
            "{bank} wale zero balance pe charge kaatenge kya",
        ],
    },
    "dbt": {
        "en": [
            "How can I check if {scheme} money has come?",
            "Where do I check my {scheme} DBT credit?",
            "How will I know {scheme} payment reached my {bank} account?",
            "How can {rel} check the {scheme} DBT in {place}?",
        ],
        "hi": [
            "{scheme} का पैसा आया या नहीं, कैसे पता करें?",
            "{scheme} की डीबीटी राशि कहाँ देखें?",
            "{bank} खाते में {scheme} का पैसा कैसे देखें?",
            "{rel} की {scheme} की राशि कैसे जाँचें?",
        ],
        "hl": [
            "{scheme} ka paisa aaya ki nahi kaise pata kare",
            "{scheme} dbt kahan check karein",
            "{bank} khate mein {scheme} ka paisa kaise dekhein",
            "{rel} ka {scheme} check karna hai kaise hoga",
        ],
    },
    "passbook": {
        "en": [
            "Where can I update my passbook?",
            "My {bank} passbook entries are pending. How do I print them?",
            "Can I get my passbook printed at Bank Mitra in {place}?",
            "{Rel}'s passbook is not updated for months. What to do?",
        ],
        "hi": [
            "पासबुक कहाँ अपडेट होगी?",
            "{bank} पासबुक में एंट्री कैसे चढ़वाएँ?",
            "{rel} की पासबुक महीनों से अपडेट नहीं हुई, क्या करें?",
        ],
        "hl": [
            "passbook kahan update hogi",
            "{bank} passbook entry kaise chadhwayein",
            "passbook print bank mitra pe hoga kya {place} mein",
            "{rel} ki passbook update nahi hui kya karein",
        ],
    },
    "free_wd": {
        "en": [
            "How many free withdrawals do I get in a Jan Dhan account?",
            "Are withdrawals free every month in my {bank} BSBDA account?",
        ],
        "hi": [
            "जन धन खाते में महीने में कितनी निकासी मुफ़्त है?",
            "{bank} के बेसिक खाते में हर महीने कितनी बार मुफ़्त निकाल सकते हैं?",
        ],
        "hl": [
            "jan dhan khate mein mahine mein kitni baar free nikal sakte hain",
            "{bank} basic khate mein free withdrawal kitne hain",
        ],
    },
    "open": {
        "en": [
            "How do I open a zero balance account for {rel}?",
            "What documents are needed to open a Jan Dhan account in {place}?",
            "Can {rel} open a {bank} account with zero balance?",
        ],
        "hi": [
            "{rel} का ज़ीरो बैलेंस खाता कैसे खुलेगा?",
            "{place} में जन धन खाता खोलने के लिए क्या चाहिए?",
        ],
        "hl": [
            "{rel} ka zero balance khata kaise khulega",
            "jan dhan khata kholna hai {place} mein kya lagega",
            "{bank} mein bina paise ke khata khulega kya",
        ],
    },
}

PASSBOOK_REPLIES = {
    "zero_bal": (
        "Jan Dhan (PMJDY/BSBDA) accounts need zero minimum balance. There is no penalty for low "
        "balance, deposits are unlimited, and you get at least 4 free withdrawals every month.",
        "जन धन (बीएसबीडीए) खाते में न्यूनतम बैलेंस ज़रूरी नहीं है। कम बैलेंस पर कोई जुर्माना नहीं, "
        "जमा असीमित है, और हर महीने कम से कम 4 निकासी मुफ़्त हैं।",
    ),
    "dbt": (
        "To check {s} DBT money, ask your Bank Mitra for an Aadhaar balance enquiry, or get your "
        "passbook printed at the branch. Carry Aadhaar and passbook.",
        "{sh} का डीबीटी पैसा देखने के लिए बैंक मित्र पर आधार से बैलेंस जाँच कराएँ, या शाखा में "
        "पासबुक प्रिंट कराएँ। आधार और पासबुक साथ रखें।",
    ),
    "passbook": (
        "Passbook printing is done at your bank branch. Carry the passbook and Aadhaar. Your Bank Mitra "
        "can also show DBT credits through an Aadhaar enquiry.",
        "पासबुक प्रिंट बैंक शाखा में होती है। पासबुक और आधार साथ ले जाएँ। बैंक मित्र भी आधार से "
        "डीबीटी जमा की जानकारी दे सकते हैं।",
    ),
    "free_wd": (
        "A BSBDA (Jan Dhan) account gives at least 4 free withdrawals every month, with zero minimum "
        "balance and unlimited deposits.",
        "बीएसबीडीए (जन धन) खाते में हर महीने कम से कम 4 निकासी मुफ़्त हैं, न्यूनतम बैलेंस शून्य है "
        "और जमा असीमित है।",
    ),
    "open": (
        "Visit the branch or Bank Mitra with Aadhaar to open a zero-balance Jan Dhan account. No minimum "
        "balance is needed and DBT money can come directly into it.",
        "ज़ीरो बैलेंस जन धन खाता खोलने के लिए आधार लेकर शाखा या बैंक मित्र जाएँ। न्यूनतम बैलेंस नहीं "
        "चाहिए और डीबीटी पैसा सीधे इसमें आ सकता है।",
    ),
}

PASSBOOK_KIND_WEIGHTS = (("zero_bal", 0.24), ("dbt", 0.30), ("passbook", 0.20), ("free_wd", 0.10), ("open", 0.16))


def gen_passbook(rng):
    reg = pick_register(rng)
    r, acc, kind = rng.random(), 0.0, "dbt"
    for k, w in PASSBOOK_KIND_WEIGHTS:
        acc += w
        if r < acc:
            kind = k
            break
    scheme = rng.choice(SCHEMES)
    idx = {"en": 0, "hi": 1, "hl": 2}[reg]
    q = fill(rng.choice(PASSBOOK_Q[kind][reg]), reg, rng, scheme=scheme[idx], small=rng.randint(0, 500))
    en, hi = PASSBOOK_REPLIES[kind]
    tgt = task_a(en.format(s=scheme[0]), hi.format(sh=scheme[1]), "PASSBOOK_DBT", 0)
    return finish_query(q, reg, rng), reg, tgt, "RBI_BSBDA_DBT_02"


# ---------------------------------------------------------------------------
# LOAN_KCC (boundary Rs 2,00,000)
# ---------------------------------------------------------------------------
KCC_Q_AMT = {
    "en": [
        "Can I get a {amt} KCC loan without collateral?",
        "I need {amt} Kisan Credit Card loan for {crop}. Will the bank ask for land papers?",
        "{amt} KCC loan documents?",
        "Is a {amt} crop loan possible on KCC without mortgage?",
        "How do I get {amt} KCC loan from {bank} in {place}?",
    ],
    "hi": [
        "क्या {amt} का केसीसी लोन बिना गिरवी मिलेगा?",
        "{crop} के लिए {amt} का किसान क्रेडिट कार्ड लोन चाहिए, क्या ज़मीन गिरवी रखनी होगी?",
        "{amt} केसीसी लोन के लिए कौन से कागज़ चाहिए?",
        "{bank} से {amt} का केसीसी लोन कैसे मिलेगा?",
    ],
    "hl": [
        "{amt} ka kcc loan bina girvi milega kya",
        "{crop} ke liye {amt} kisan card loan chahiye zameen girvi rakhni padegi kya",
        "{amt} kcc loan ke kagaz kya lagenge",
        "{bank} se {amt} kcc kaise milega {place} mein",
    ],
}
KCC_Q_DOCS = {
    "en": [
        "What documents are needed for a Kisan Credit Card?",
        "How do I apply for KCC at {bank}?",
        "How much KCC loan can I get without mortgaging land for {crop} farming?",
        "Where do I apply for a Kisan Credit Card in {place}?",
    ],
    "hi": [
        "बिना ज़मीन गिरवी रखे कितना केसीसी लोन मिलेगा?",
        "किसान क्रेडिट कार्ड के लिए कौन से दस्तावेज़ चाहिए?",
        "{bank} में केसीसी के लिए कैसे आवेदन करें?",
        "{crop} की खेती के लिए किसान क्रेडिट कार्ड कैसे बनेगा?",
    ],
    "hl": [
        "kcc ke liye kya kya document lagta hai",
        "bina zameen girvi kitna kcc milega",
        "{bank} mein kisan credit card kaise banega",
        "{crop} ki kheti ke liye kcc kaise banega {place} mein",
    ],
}
KCC_LE_2L = (
    "KCC crop loans up to Rs 2,00,000 are collateral-free, so Rs {a} needs no land mortgage. Apply at "
    "your bank branch with Khatauni, 2 Aadhaar/PAN copies, passbook and 3 photos.",
    "₹2,00,000 तक का केसीसी फसल ऋण बिना गिरवी मिलता है, इसलिए ₹{a} के लिए ज़मीन गिरवी नहीं चाहिए। "
    "खतौनी, आधार/पैन की 2 प्रतियाँ, पासबुक और 3 फ़ोटो लेकर शाखा जाएँ।",
)
KCC_GT_2L = (
    "Only KCC loans up to Rs 2,00,000 are collateral-free. For Rs {a}, the branch may ask for "
    "security. Visit the branch with Khatauni, 2 Aadhaar/PAN copies, passbook and 3 photos.",
    "केवल ₹2,00,000 तक का केसीसी ऋण बिना गिरवी है। ₹{a} के लिए शाखा गिरवी या सुरक्षा माँग सकती है। "
    "खतौनी, आधार/पैन की 2 प्रतियाँ, पासबुक और 3 फ़ोटो लेकर शाखा जाएँ।",
)
KCC_DOCS = (
    "Apply for a Kisan Credit Card at your bank branch. Carry Khatauni, 2 Aadhaar/PAN copies, "
    "passbook and 3 photos. Crop loans up to Rs 2,00,000 need no collateral.",
    "किसान क्रेडिट कार्ड के लिए बैंक शाखा में आवेदन करें। खतौनी, आधार/पैन की 2 प्रतियाँ, पासबुक और "
    "3 फ़ोटो ले जाएँ। ₹2,00,000 तक का फसल ऋण बिना गिरवी मिलता है।",
)


def amount_kcc(rng):
    r = rng.random()
    if r < 0.2:
        return rng.choice([199999, 200000, 200001, 250000])
    if r < 0.6:
        return rng.choice([25000, 50000, 75000, 100000, 120000, 150000, 160000, 175000, 180000])
    if r < 0.8:
        return rng.choice([210000, 300000, 350000, 400000, 500000])
    return rng.randrange(10, 400) * 1000


def gen_kcc(rng):
    reg = pick_register(rng)
    if rng.random() < 0.25:
        q = fill(rng.choice(KCC_Q_DOCS[reg]), reg, rng)
        en, hi = KCC_DOCS
        tgt = task_a(en, hi, "LOAN_KCC", 0)
    else:
        n = amount_kcc(rng)
        q = fill(rng.choice(KCC_Q_AMT[reg]), reg, rng, amt=fmt_amt(reg, n, rng))
        en, hi = KCC_LE_2L if n <= 200000 else KCC_GT_2L
        tgt = task_a(en.format(a=inr(n)), hi.format(a=inr(n)), "LOAN_KCC", n)
    return finish_query(q, reg, rng), reg, tgt, "RBI_KCC_MASTER_01"


# ---------------------------------------------------------------------------
# SAFE ABSTENTION ("never guess") traps (guide page 5)
# ---------------------------------------------------------------------------
ABSTAIN_Q = {
    "live": {
        "en": [
            "What is the exact balance in my {bank} account {acct} right now?",
            "Did my Rs {amt} NEFT to {rel} clear?",
            "I forgot my ATM PIN. Tell me my PIN.",
            "How much money is in my Jan Dhan account today?",
            "Has my {scheme} instalment come into my account today?",
            "What is my loan outstanding at {bank}?",
            "Tell me my {bank} account number.",
            "When will my cheque of Rs {amt} clear?",
            "Exact balance in {bank} acct {acct}?",
            "Is my KCC loan of Rs {amt} sanctioned yet?",
        ],
        "hi": [
            "मेरे {bank} खाते {acct} में अभी कितना बैलेंस है?",
            "क्या मेरा ₹{amt} का एनईएफ़टी {rel} तक पहुँच गया?",
            "मेरा एटीएम पिन भूल गया, बता दो।",
            "आज मेरे जन धन खाते में कितने पैसे हैं?",
            "क्या मेरी {scheme} की किस्त आज खाते में आई?",
            "{bank} में मेरा कितना लोन बाकी है?",
            "मेरा ₹{amt} का चेक कब क्लियर होगा?",
            "क्या मेरा ₹{amt} का केसीसी लोन पास हो गया?",
        ],
        "hl": [
            "mere {bank} khate mein abhi kitna balance hai",
            "mera {amt} ka neft {rel} ko gaya ki nahi",
            "atm pin bhul gaya batao na",
            "{scheme} ka paisa aaya kya aaj",
            "mera loan kitna baaki hai {bank} mein",
            "khata number {acct} mein kitne paise hain",
            "mera {amt} ka cheque kab clear hoga",
            "kcc loan {amt} ka pass hua kya",
        ],
    },
    "external": {
        "en": [
            "What is the wheat price in {place} mandi today?",
            "Which medicine should I take for fever?",
            "What is the HDFC share price today?",
            "Will it rain in {place} tomorrow?",
            "Which fertilizer is best for {crop}?",
            "What is the {crop} rate in {place} mandi?",
            "Gold rate in {place} today?",
            "Which seed variety of {crop} gives the most yield?",
        ],
        "hi": [
            "आज {place} मंडी में गेहूँ का भाव क्या है?",
            "बुखार में कौन सी दवा लूँ?",
            "कल {place} में बारिश होगी क्या?",
            "{crop} के लिए सबसे अच्छी खाद कौन सी है?",
            "{place} मंडी में {crop} का रेट क्या है?",
            "आज सोने का भाव क्या है?",
        ],
        "hl": [
            "{place} mandi mein gehun ka bhav kya hai",
            "bukhar ki dawai batao",
            "kal {place} mein baarish hogi kya",
            "{crop} ke liye kaun si khaad daalein",
            "{crop} ka rate kya chal raha hai {place} mein",
            "share market mein kya lagau",
        ],
    },
    "rumor": {
        "en": [
            "Will PM-Kisan double to Rs 12,000 this year?",
            "Is there a new RBI loan waiver announced 10 minutes ago?",
            "Will the government give Rs {amt} to every farmer next month?",
            "Is {bank} going to shut down next week?",
            "Is the KCC limit increasing to Rs 5 lakh from tomorrow?",
            "Will Jan Dhan accounts get Rs {amt} free deposit?",
            "My neighbour says all {bank} loans in {place} are waived. Is it true?",
        ],
        "hi": [
            "क्या इस साल पीएम-किसान ₹12,000 हो जाएगा?",
            "क्या आरबीआई ने अभी 10 मिनट पहले कर्ज़ माफ़ी की घोषणा की?",
            "क्या सरकार अगले महीने हर किसान को ₹{amt} देगी?",
            "क्या {bank} अगले हफ़्ते बंद हो जाएगा?",
            "क्या जन धन खाते में ₹{amt} मुफ़्त आएँगे?",
            "पड़ोसी कह रहे हैं {place} में सबका लोन माफ़ हो गया, सच है?",
        ],
        "hl": [
            "pm kisan 12000 ho jayega kya is saal",
            "rbi ne abhi karza maafi ki kya",
            "sarkar har kisan ko {amt} degi kya agle mahine",
            "{bank} band hone wala hai kya",
            "jan dhan mein {amt} free aayenge kya",
            "{place} mein sabka loan maaf ho gaya kya",
        ],
    },
}

ABSTAIN_REPLIES = {
    "live": (
        "I cannot check live balances, PINs or transaction status offline. Please visit your bank "
        "branch with your passbook, or call the RBI helpline 14448 for help.",
        "मैं ऑफ़लाइन लाइव बैलेंस, पिन या लेनदेन की स्थिति नहीं देख सकता। कृपया पासबुक लेकर बैंक शाखा "
        "जाएँ या आरबीआई हेल्पलाइन 14448 पर कॉल करें।",
    ),
    "external": (
        "This is outside verified banking rules, so I cannot answer it. For banking help, visit your "
        "branch or call the RBI helpline 14448.",
        "यह सत्यापित बैंकिंग नियमों से बाहर है, इसलिए मैं इसका जवाब नहीं दे सकता। बैंकिंग मदद के लिए "
        "शाखा जाएँ या आरबीआई हेल्पलाइन 14448 पर कॉल करें।",
    ),
    "rumor": (
        "I never guess about new schemes or rumours. Please check at your bank branch or RBI helpline "
        "14448. If anyone asks for money or OTP for a scheme, call 1930.",
        "मैं नई योजनाओं या अफ़वाहों पर अंदाज़ा नहीं लगाता। कृपया बैंक शाखा या आरबीआई हेल्पलाइन 14448 "
        "से पुष्टि करें। अगर कोई योजना के नाम पर पैसे या ओटीपी माँगे, तो 1930 पर कॉल करें।",
    ),
}
ABSTAIN_SOURCES = {
    "live": ["RBI_BSBDA_DBT_02", "RBI_BC_BSBDA_01", "RBI_BC_TRANSFER_03"],
    "external": ["RBI_KCC_MASTER_01", "RBI_BSBDA_DBT_02"],
    "rumor": ["RBI_KCC_MASTER_01", "RBI_BSBDA_DBT_02"],
}


def gen_abstain(rng):
    reg = pick_register(rng)
    kind = rng.choices(["live", "external", "rumor"], weights=[0.45, 0.30, 0.25])[0]
    n = rng.choice([500, 1000, 2000, 4000, 5000, 6000, 10000, 12000, 15000, 25000, 50000])
    amt = inr(n) if reg != "hl" else str(n)
    q = fill(rng.choice(ABSTAIN_Q[kind][reg]), reg, rng, acct=acct_no(rng), amt=amt)
    en, hi = ABSTAIN_REPLIES[kind]
    tgt = task_a(en, hi, "NONE", 0, abstained=True)
    return finish_query(q, reg, rng), reg, tgt, rng.choice(ABSTAIN_SOURCES[kind])


# ---------------------------------------------------------------------------
# ASSISTANT SCAM ALERT (Task A, isScamAlert=true)
# ---------------------------------------------------------------------------
SCAM_ALERT_Q = {
    "generic": {
        "en": [
            "A caller from {bank} asked for the OTP sent to my phone. Should I tell him?",
            "Someone says my KCC subsidy will come if I share my ATM PIN.",
            "Caller wants my card CVV to stop my {bank} card from blocking.",
            "Caller asked OTP and to install {app}.",
            "Bank officer on phone says install {app} to get refund of Rs {amt}.",
            "WhatsApp sent me a {bank}_KYC.apk file for {scheme} money. Should I install it?",
            "I won a lottery of Rs {amt}. They ask my UPI PIN to receive it.",
            "Man on phone says he is from RBI and needs my debit card number and OTP for {scheme}.",
        ],
        "hi": [
            "{bank} से फ़ोन आया, मेरे फ़ोन पर आया ओटीपी माँग रहे हैं, बता दूँ?",
            "कोई कह रहा है एटीएम पिन बताओ तो केसीसी सब्सिडी आएगी।",
            "कॉल करने वाले ने ओटीपी माँगा और {app} डालने को कहा।",
            "फ़ोन पर बैंक अधिकारी ₹{amt} रिफ़ंड के लिए {app} ऐप डालने को कह रहा है।",
            "व्हाट्सऐप पर {scheme} के पैसे के लिए एक एपीके फ़ाइल आई है, डाल लूँ?",
            "₹{amt} की लॉटरी लगी है, पैसे पाने के लिए यूपीआई पिन माँग रहे हैं।",
        ],
        "hl": [
            "{bank} se call aaya otp maang rahe hain bata du kya",
            "koi bol raha atm pin batao to kcc subsidy aayegi",
            "caller ne otp manga aur {app} download karne bola",
            "phone pe bank wala bol raha {app} dalo {amt} refund milega",
            "whatsapp pe apk file aayi hai {scheme} ke paise ke liye install karu kya",
            "lottery lagi hai {amt} ki upi pin maang rahe hain",
            "kyc ke naam pe card ka cvv maang raha hai",
        ],
    },
    "link": {
        "en": [
            "I got an SMS with a bit.ly link to update KYC, else my {bank} account will be blocked.",
            "Message says click tinyurl link to unblock my {bank} account today.",
        ],
        "hi": [
            "एसएमएस आया है कि केवाईसी के लिए लिंक खोलो वरना {bank} खाता बंद हो जाएगा।",
            "मैसेज में लिंक है, कह रहे हैं आज ही क्लिक करो वरना खाता ब्लॉक।",
        ],
        "hl": [
            "kyc update ka link aaya hai warna {bank} khata band ho jayega",
            "bitly link pe click karne bola account unblock ke liye",
        ],
    },
    "double_thumb": {
        "en": [
            "Bank Mitra took my thumb twice and gave no receipt.",
            "BC agent in {place} said server failed and scanned my thumb again without a receipt.",
        ],
        "hi": [
            "बैंक मित्र ने दो बार अंगूठा लगवाया और रसीद नहीं दी।",
            "{place} में बीसी वाले ने सर्वर फ़ेल बोलकर दोबारा अंगूठा लगवाया, रसीद नहीं दी।",
        ],
        "hl": [
            "bank mitra ne do baar angootha lagwaya receipt nahi di",
            "bc wale ne server fail bolke dobara thumb lagwaya {place} mein",
        ],
    },
}

SCAM_ALERT_REPLIES = {
    "generic": (
        "Stop, this is a scam. Never share OTP, PIN or CVV and never install AnyDesk or unknown apps. "
        "Call cyber helpline 1930 now and inform your bank within 3 working days for zero liability.",
        "रुकिए, यह धोखाधड़ी है। ओटीपी, पिन या सीवीवी कभी न बताएँ और एनीडेस्क या अनजान ऐप न डालें। "
        "तुरंत 1930 पर कॉल करें और 3 कार्य दिवस में बैंक को बताएँ।",
    ),
    "link": (
        "Do not open this link. Banks never ask for KYC through bit.ly links, OTP or PIN. Call 1930 now "
        "and inform your bank within 3 working days to stay protected.",
        "यह लिंक न खोलें। बैंक कभी लिंक, ओटीपी या पिन से केवाईसी नहीं माँगता। तुरंत 1930 पर कॉल करें "
        "और 3 कार्य दिवस में बैंक को बताएँ।",
    ),
    "double_thumb": (
        "This is a warning sign of AePS fraud. Never give your thumb twice without a printed receipt. "
        "Call 1930 now and report to your bank within 3 working days for zero liability.",
        "यह एईपीएस धोखाधड़ी का संकेत है। बिना छपी रसीद के दोबारा अंगूठा न लगाएँ। तुरंत 1930 पर कॉल "
        "करें और 3 कार्य दिवस में बैंक को बताएँ।",
    ),
}


def gen_scam_alert(rng):
    reg = pick_register(rng)
    kind = rng.choices(["generic", "link", "double_thumb"], weights=[0.65, 0.2, 0.15])[0]
    n = rng.choice([999, 2500, 5000, 10000, 25000, 50000, 100000, 500000])
    amt = inr(n) if reg != "hl" else str(n)
    app = rng.choice(REMOTE_APPS)
    q = fill(rng.choice(SCAM_ALERT_Q[kind][reg]), reg, rng, amt=amt,
             app=REMOTE_APPS_HI[app] if reg == "hi" else app)
    en, hi = SCAM_ALERT_REPLIES[kind]
    tgt = task_a(en, hi, "NONE", 0, scam=True)
    return finish_query(q, reg, rng), reg, tgt, "RBI_ZERO_LIABILITY_1930"


# ---------------------------------------------------------------------------
# Task B: ScamGuard SMS / call analysis (3 tiers)
# ---------------------------------------------------------------------------
HIGH_RISK_TIP = (
    "Do not reply, click or share OTP. Call 1930 now and report to your bank within 3 working days "
    "for zero liability.",
    "जवाब न दें, लिंक न खोलें, ओटीपी न बताएँ। तुरंत 1930 पर कॉल करें और 3 कार्य दिवस में बैंक को "
    "बताएँ।",
)
HIGH_RISK_EXPL = {
    "link": (
        "This message pushes a short KYC link and threatens account blocking. Real banks never ask for "
        "KYC, OTP or PIN through links.",
        "यह संदेश छोटा केवाईसी लिंक भेजकर खाता बंद करने की धमकी देता है। असली बैंक कभी लिंक से "
        "केवाईसी, ओटीपी या पिन नहीं माँगता।",
    ),
    "otp": (
        "The caller is asking for OTP, PIN or CVV. No bank or government officer ever asks for these.",
        "कॉल करने वाला ओटीपी, पिन या सीवीवी माँग रहा है। कोई बैंक या सरकारी अधिकारी ये कभी नहीं माँगता।",
    ),
    "remote": (
        "Screen sharing apps like AnyDesk let fraudsters control your phone and empty your account.",
        "एनीडेस्क जैसे स्क्रीन शेयरिंग ऐप से ठग आपका फ़ोन चलाकर खाता खाली कर सकते हैं।",
    ),
    "apk": (
        "An .apk file sent on WhatsApp can secretly read your OTPs and bank SMS.",
        "व्हाट्सऐप पर भेजी गई एपीके फ़ाइल चुपके से आपके ओटीपी और बैंक एसएमएस पढ़ सकती है।",
    ),
    "lottery": (
        "Prize messages asking for UPI PIN are fraud. A PIN is needed only to send money, never to "
        "receive it.",
        "इनाम के नाम पर यूपीआई पिन माँगना धोखाधड़ी है। पिन सिर्फ़ पैसे भेजने में लगता है, पाने में कभी नहीं।",
    ),
    "thumb": (
        "Taking your thumb twice without a printed receipt can hide a second withdrawal from your account.",
        "बिना छपी रसीद के दो बार अंगूठा लगवाना आपके खाते से छिपी हुई दूसरी निकासी हो सकती है।",
    ),
}
HIGH_RISK_MSGS = [
    ("link", "Dear {bank} customer, your account will be blocked today. Update KYC now: bit.ly/{code}"),
    ("link", "{bank} ALERT: Your PAN/Aadhaar is not linked. Click tinyurl.com/{code} within 24 hrs or account will be suspended."),
    ("link", "aapka {bank} khata band ho jayega turant kyc karein bit.ly/{code}"),
    ("link", "प्रिय ग्राहक, आपका {bank_hi} खाता आज बंद हो जाएगा। केवाईसी के लिए bit.ly/{code} पर क्लिक करें।"),
    ("link", "Your {scheme} payment of Rs {amt} is on hold. Verify at http://{code}-kyc.in to release."),
    ("otp", "Call from {phone}: caller says he is from {bank} head office and asks for the OTP sent to my phone to stop card blocking."),
    ("otp", "Caller says my {scheme} money is stuck and asks my ATM PIN and CVV to release Rs {amt}."),
    ("otp", "caller bol raha hai {bank} se hoon otp batao warna card band"),
    ("remote", "Caller asked me to install {app} and read the 9-digit code to get KCC subsidy of Rs {amt}."),
    ("remote", "Customer care number {phone} told me to download {app} to get refund of Rs {amt}."),
    ("apk", "WhatsApp message with file {bank}_KYC_Update.apk asking to install for {scheme} payment."),
    ("apk", "व्हाट्सऐप पर {scheme_hi} के लिए {code}.apk फ़ाइल आई है, इंस्टॉल करने को कहा है।"),
    ("lottery", "Congratulations! You won Rs {amt} in Lucky Draw. Enter UPI PIN to receive the amount. Call {phone}."),
    ("lottery", "aapne {amt} rupaye jeete hain paise lene ke liye upi pin daalein"),
    ("thumb", "Bank Mitra in {place} put my thumb twice on the machine and said server failed, no receipt given."),
]

CAUTION_TIP = (
    "Do not pay any unofficial fee. Ask for a printed receipt and verify at your branch or RBI helpline "
    "14448.",
    "कोई अनाधिकृत फ़ीस न दें। छपी रसीद माँगें और शाखा या आरबीआई हेल्पलाइन 14448 पर पुष्टि करें।",
)
CAUTION_EXPL = {
    "fee": (
        "Bank Mitra services carry no extra cash fee. Asking for extra money is not allowed.",
        "बैंक मित्र की सेवाओं पर कोई अतिरिक्त नकद फ़ीस नहीं लगती। अतिरिक्त पैसे माँगना गलत है।",
    ),
    "loan": (
        "Urgent pre-approved loan offers from private agents are a common trap. Banks do not ask you to "
        "call agents tonight.",
        "निजी एजेंट के जल्दबाज़ी वाले प्री-अप्रूव्ड लोन ऑफ़र आम जाल हैं। बैंक रात में एजेंट को कॉल करने को "
        "नहीं कहता।",
    ),
    "advance": (
        "Asking for an advance registration fee for a Mudra or KCC loan is a warning sign. Banks process "
        "these at the branch.",
        "मुद्रा या केसीसी लोन के लिए पहले से रजिस्ट्रेशन फ़ीस माँगना खतरे का संकेत है। बैंक यह काम शाखा "
        "में करता है।",
    ),
    "subsidy": (
        "This subsidy link is not from an official source. It has not asked for OTP yet, but do not "
        "enter your details.",
        "यह सब्सिडी लिंक किसी आधिकारिक स्रोत से नहीं है। अभी ओटीपी नहीं माँगा है, फिर भी अपनी जानकारी न "
        "भरें।",
    ),
}
CAUTION_MSGS = [
    ("fee", "Bank Mitra asked Rs {fee} extra cash for withdrawing Rs {amt} with Aadhaar."),
    ("fee", "bank mitra {fee} rupaye extra maang raha hai {amt} nikalne ke"),
    ("fee", "बैंक मित्र ₹{amt} निकालने पर ₹{fee} अलग से माँग रहा है।"),
    ("fee", "BC agent in {place} says Rs {fee} tip is compulsory for DBT withdrawal."),
    ("loan", "SMS: Your pre-approved loan of Rs {amt} expires tonight. Call agent {phone} to claim."),
    ("loan", "aapka {amt} ka loan pre-approved hai aaj raat tak agent {phone} ko call karein"),
    ("advance", "Agent asking Rs 999 registration fee in advance for {loan} loan approval of Rs {amt}."),
    ("advance", "{loan} लोन के लिए एजेंट पहले ₹999 रजिस्ट्रेशन फ़ीस माँग रहा है।"),
    ("subsidy", "Message: Apply for {scheme} subsidy at {code}-yojana.online form. Fill your details."),
    ("subsidy", "{scheme} subsidy form bharo is link pe {code}-sarkari.in"),
]

SAFE_TIP = (
    "No action needed. Match the amount in your passbook on your next branch visit. Never share OTP.",
    "कुछ करने की ज़रूरत नहीं। अगली बार शाखा जाने पर पासबुक में राशि मिला लें। ओटीपी कभी न बताएँ।",
)
SAFE_EXPL = {
    "credit": (
        "This looks like a genuine credit alert. It has no link and asks for no OTP, PIN or payment.",
        "यह असली जमा सूचना लगती है। इसमें कोई लिंक नहीं है और न ही ओटीपी, पिन या भुगतान माँगा गया है।",
    ),
    "atm": (
        "This is a normal withdrawal alert. It has no link and asks for nothing. If you did not withdraw, "
        "inform your branch.",
        "यह सामान्य निकासी सूचना है। इसमें कोई लिंक नहीं है और कुछ नहीं माँगा गया। अगर निकासी आपने नहीं "
        "की, तो शाखा को बताएँ।",
    ),
    "kyc": (
        "This is a genuine branch KYC reminder. It asks you to visit the branch, not to click a link or "
        "share OTP.",
        "यह शाखा की असली केवाईसी सूचना है। इसमें शाखा जाने को कहा है, लिंक खोलने या ओटीपी बताने को नहीं।",
    ),
}
SAFE_MSGS = [
    ("credit", "Rs {amt}.00 credited to A/c XX{l4} on {date} by PM-KISAN DBT. Avl Bal Rs {bal}. -{bank}"),
    ("credit", "आपके खाते XX{l4} में मनरेगा मज़दूरी ₹{amt} जमा हुई है। -{bank_hi}"),
    ("credit", "A/c XX{l4} credited with Rs {amt} LPG subsidy (DBTL) on {date}. -{bank}"),
    ("credit", "Dear Customer, Rs {amt} {scheme} credited to your account XX{l4}. -{bank}"),
    ("credit", "आपके खाते XX{l4} में {scheme_hi} की राशि ₹{amt} दिनांक {date} को जमा हुई। -{bank_hi}"),
    ("atm", "Rs {amt} withdrawn from A/c XX{l4} at ATM on {date}. If not done by you, contact your branch. -{bank}"),
    ("kyc", "Dear customer, please visit your home branch with Aadhaar to update KYC. {bank} never asks for OTP or PIN. -{bank}"),
    ("kyc", "प्रिय ग्राहक, केवाईसी अपडेट के लिए आधार लेकर अपनी शाखा पधारें। {bank_hi} कभी ओटीपी नहीं माँगता।"),
]


def _scam_slots(rng):
    bank = rng.choice(BANKS)
    scheme = rng.choice(SCHEMES)
    return {
        "bank": bank[0], "bank_hi": bank[1], "scheme": scheme[0], "scheme_hi": scheme[1],
        "code": short_code(rng), "phone": phone(rng), "app": rng.choice(REMOTE_APPS),
        "place": rng.choice(PLACES)[0], "l4": last4(rng), "date": sms_date(rng),
        "fee": rng.choice([20, 30, 40, 50]), "loan": rng.choice(["Mudra", "KCC", "PM Svanidhi"]),
        "bal": inr(rng.randint(2000, 60000)),
    }


def gen_scam_b(rng, level: str):
    s = _scam_slots(rng)
    if level == "HIGH_RISK":
        kind, tmpl = rng.choice(HIGH_RISK_MSGS)
        s["amt"] = inr(rng.choice([999, 5000, 9999, 25000, 49999, 100000, 250000, 1000000]))
        exp, tip = HIGH_RISK_EXPL[kind], HIGH_RISK_TIP
    elif level == "CAUTION":
        kind, tmpl = rng.choice(CAUTION_MSGS)
        s["amt"] = inr(rng.choice([2000, 3000, 5000, 6000, 8000, 50000, 100000, 200000]))
        exp, tip = CAUTION_EXPL[kind], CAUTION_TIP
    else:
        kind, tmpl = rng.choice(SAFE_MSGS)
        s["amt"] = inr(rng.choice([2000, 2000, 2000, 4000, 6000, 237, 1450, 3120, 1500, 3000]))
        exp, tip = SAFE_EXPL[kind], SAFE_TIP
    msg = tmpl.format(**s)
    reg = "hi" if any("ऀ" <= ch <= "ॿ" for ch in msg) else ("hl" if msg[:1].islower() else "en")
    tgt = task_b(level, exp[0], exp[1], tip[0], tip[1])
    return msg, reg, tgt


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------
GENERATORS_A = {
    "loan_kcc": gen_kcc,
    "passbook_dbt_bsbda": gen_passbook,
    "transfer_ippb_bc": gen_transfer,
    "safe_abstention": gen_abstain,
    "withdraw_bc_le_10k": lambda rng: gen_withdraw(rng, False),
    "withdraw_branch_gt_10k": lambda rng: gen_withdraw(rng, True),
    "assistant_scam_alert": gen_scam_alert,
}
GENERATORS_B = {
    "scam_high_risk": lambda rng: gen_scam_b(rng, "HIGH_RISK"),
    "scam_caution": lambda rng: gen_scam_b(rng, "CAUTION"),
    "scam_safe": lambda rng: gen_scam_b(rng, "SAFE"),
}


def make_row(bucket: str, rng: random.Random) -> dict:
    if bucket in GENERATORS_A:
        query, reg, target, src = GENERATORS_A[bucket](rng)
        grounded = rng.random() < GROUNDED_RATE
        ctx: list[str] = []
        if grounded:
            ctx = [src]
            if rng.random() < DISTRACTOR_RATE:
                ctx.append(rng.choice([s for s in SOURCES if s != src]))
                rng.shuffle(ctx)
        prompt = build_task_a_prompt(query, ctx)
        errs = validate_task_a(target)
        hi_ok = devanagari_ok(target["replyHi"])
        task_type = "assistant"
    else:
        query, reg, target = GENERATORS_B[bucket](rng)
        grounded, ctx = False, []
        prompt = build_task_b_prompt(query)
        errs = validate_task_b(target)
        hi_ok = devanagari_ok(target["explanationHi"]) and devanagari_ok(target["actionTipHi"])
        task_type = "scam"
    if errs or not hi_ok:
        raise ValueError(f"[{bucket}] invalid target {errs} hi_ok={hi_ok}: {target}")
    completion = compact_json(target)
    return {
        "bucket": bucket,
        "task_type": task_type,
        "register": reg,
        "grounded": grounded,
        "context_ids": ctx,
        "query": query,
        "prompt": prompt,
        "completion": completion,
        "text": wrap_full(prompt, completion),
        "query_sha256": query_hash(query),
    }


def compute_quotas(target_rows: int) -> dict[str, int]:
    raw = {b: target_rows * w for b, w in BUCKET_WEIGHTS.items()}
    quotas = {b: int(v) for b, v in raw.items()}
    remainder = target_rows - sum(quotas.values())
    for b in sorted(raw, key=lambda k: raw[k] - quotas[k], reverse=True)[:remainder]:
        quotas[b] += 1
    return quotas


def generate(target_rows: int, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    seen: set[str] = set()
    by_bucket: dict[str, list[dict]] = {}
    for bucket, quota in compute_quotas(target_rows).items():
        rows, attempts = [], 0
        while len(rows) < quota and attempts < quota * MAX_ATTEMPTS_PER_ROW:
            attempts += 1
            row = make_row(bucket, rng)
            if row["query_sha256"] in seen:
                continue
            seen.add(row["query_sha256"])
            rows.append(row)
        if len(rows) < quota:
            print(f"  note: {bucket} reached {len(rows)}/{quota} unique rows (dedupe saturation)")
        by_bucket[bucket] = rows
    return by_bucket


def split(by_bucket: dict[str, list[dict]], eval_fraction: float, seed: int):
    rng = random.Random(seed + 1)
    train, evals = [], []
    for bucket, rows in by_bucket.items():
        rows = rows[:]
        rng.shuffle(rows)
        n_eval = max(1, round(len(rows) * eval_fraction))
        evals += rows[:n_eval]
        train += rows[n_eval:]
    rng.shuffle(train)
    rng.shuffle(evals)
    for i, r in enumerate(train):
        r["id"] = f"bs-train-{i:05d}"
    for i, r in enumerate(evals):
        r["id"] = f"bs-eval-{i:05d}"
    return train, evals


def summarize(train: list[dict], evals: list[dict]) -> dict:
    all_rows = train + evals
    total = len(all_rows)

    def dist(key):
        c = collections.Counter(r[key] for r in all_rows)
        return {k: {"rows": v, "pct": round(100 * v / total, 1)} for k, v in c.most_common()}

    task_a_rows = [r for r in all_rows if r["task_type"] == "assistant"]
    leakage = len({r["query_sha256"] for r in train} & {r["query_sha256"] for r in evals})
    return {
        "total_rows": total,
        "train_rows": len(train),
        "eval_rows": len(evals),
        "train_eval_leakage_count": leakage,
        "buckets": dist("bucket"),
        "registers": dist("register"),
        "grounded_pct_task_a": round(100 * sum(r["grounded"] for r in task_a_rows) / max(1, len(task_a_rows)), 1),
        "withdraw_share_pct": round(100 * sum(r["bucket"].startswith("withdraw_") for r in all_rows) / total, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--target-rows", type=int, default=1500, help="1,000-3,000 recommended")
    ap.add_argument("--eval-fraction", type=float, default=EVAL_FRACTION)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)

    if not 1000 <= args.target_rows <= 3000:
        print(f"warning: --target-rows {args.target_rows} is outside the validated 1,000-3,000 range")

    by_bucket = generate(args.target_rows, args.seed)
    train, evals = split(by_bucket, args.eval_fraction, args.seed)
    stats = summarize(train, evals)
    if stats["train_eval_leakage_count"] != 0:
        print("ERROR: train/eval leakage detected", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "eval.jsonl"), evals)
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Wrote {stats['total_rows']} deduplicated rows -> {stats['train_rows']} train / {stats['eval_rows']} eval")
    for b, v in stats["buckets"].items():
        print(f"  {b:<24} {v['rows']:>5} rows ({v['pct']}%)")
    print("  registers: " + ", ".join(f"{k}={v['pct']}%" for k, v in stats["registers"].items()))
    print(f"  grounded (Task A): {stats['grounded_pct_task_a']}%   leakage: {stats['train_eval_leakage_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

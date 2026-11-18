#!/usr/bin/env python3
"""BankSathi 5-KPI release gate (guide pages 1, 3, 7).

Three modes:

1. Pre-GPU dataset audit (no model):
     python3 eval_gemma_banksathi.py --train-file data/train.jsonl --eval-file data/eval.jsonl
   Checks valid JSON, schema, <=45-word TTS caps, Devanagari parity, SHA-256 train/eval leakage,
   class balance. Requires valid_json_pct == 100 and devanagari_parity_pct == 100.

2. Live model benchmark on the hold-out set:
     python3 eval_gemma_banksathi.py --model-dir outputs/gemma2-2b-banksathi/merged_fp16 --eval-file data/eval.jsonl
   Greedy-decodes every eval prompt and scores the 5 KPIs:
     valid_json_pct >= 99 | intent_accuracy_pct >= 96 | scamguard_recall_pct == 100
     safe_abstention_pct >= 95 | bilingual_tts_pct >= 98

3. Score saved raw outputs (e.g. from the phone or a previous run):
     python3 eval_gemma_banksathi.py --eval-file data/eval.jsonl --predictions-file preds.jsonl
   preds.jsonl: one {"id": ..., "output": "<raw model text>"} per line.

Exit code is 0 only when every gate passes.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys

from banksathi_common import (
    END_TURN,
    REPLY_WORD_CAP,
    EXPLANATION_WORD_CAP,
    devanagari_ok,
    read_jsonl,
    validate_task_a,
    validate_task_b,
    word_count,
    wrap_prompt,
)

KPI_GATES = {
    "valid_json_pct": 99.0,
    "intent_accuracy_pct": 96.0,
    "scamguard_recall_pct": 100.0,
    "safe_abstention_pct": 95.0,
    "bilingual_tts_pct": 98.0,
}


def pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 2) if den else 100.0


# ---------------------------------------------------------------------------
# Mode 1: dataset audit
# ---------------------------------------------------------------------------
def audit_dataset(train: list[dict], evals: list[dict]) -> dict:
    rows = train + evals
    valid_json = parity = caps = 0
    problems = []
    for r in rows:
        try:
            obj = json.loads(r["completion"])
        except json.JSONDecodeError as e:
            problems.append((r.get("id"), f"json: {e}"))
            continue
        validator = validate_task_a if r["task_type"] == "assistant" else validate_task_b
        schema_errs = validator(obj, strict_caps=False)
        if schema_errs:
            problems.append((r.get("id"), f"schema: {schema_errs}"))
            continue
        valid_json += 1
        if r["task_type"] == "assistant":
            hi_ok = devanagari_ok(obj["replyHi"])
            cap_ok = word_count(obj["replyEn"]) <= REPLY_WORD_CAP and word_count(obj["replyHi"]) <= REPLY_WORD_CAP
        else:
            hi_ok = devanagari_ok(obj["explanationHi"]) and devanagari_ok(obj["actionTipHi"])
            cap_ok = all(word_count(obj[k]) <= EXPLANATION_WORD_CAP
                         for k in ("explanationEn", "explanationHi", "actionTipEn", "actionTipHi"))
        parity += hi_ok
        caps += cap_ok
        if not hi_ok:
            problems.append((r.get("id"), "Devanagari parity"))
        if not cap_ok:
            problems.append((r.get("id"), "word cap"))
        if not r["text"].endswith(r["completion"] + END_TURN) or not r["text"].startswith(wrap_prompt(r["prompt"])):
            problems.append((r.get("id"), "chat wrapper mismatch"))

    train_h = {r["query_sha256"] for r in train}
    eval_h = {r["query_sha256"] for r in evals}
    dup_within = len(rows) - len({r["query_sha256"] for r in rows})
    buckets = collections.Counter(r["bucket"] for r in rows)
    total = len(rows)
    withdraw_share = pct(sum(v for k, v in buckets.items() if k.startswith("withdraw_")), total)
    max_bucket_share = max(pct(v, total) for v in buckets.values()) if buckets else 0.0

    report = {
        "mode": "dataset_audit",
        "total_rows": total,
        "train_rows": len(train),
        "eval_rows": len(evals),
        "valid_json_pct": pct(valid_json, total),
        "devanagari_parity_pct": pct(parity, total),
        "tts_word_cap_pct": pct(caps, total),
        "train_eval_leakage_count": len(train_h & eval_h),
        "duplicate_query_count": dup_within,
        "withdraw_share_pct": withdraw_share,
        "max_bucket_share_pct": max_bucket_share,
        "buckets": dict(buckets.most_common()),
        "registers": dict(collections.Counter(r["register"] for r in rows).most_common()),
        "problems_sample": problems[:20],
    }
    report["gates"] = {
        "valid_json_pct == 100": report["valid_json_pct"] == 100.0,
        "devanagari_parity_pct == 100": report["devanagari_parity_pct"] == 100.0,
        "tts_word_cap_pct == 100": report["tts_word_cap_pct"] == 100.0,
        "train_eval_leakage_count == 0": report["train_eval_leakage_count"] == 0,
        "duplicate_query_count == 0": dup_within == 0,
        "withdraw_share_pct <= 25": withdraw_share <= 25.0,
        "max_bucket_share_pct <= 16.5": max_bucket_share <= 16.5,
        "no wrapper problems": not any(p[1] == "chat wrapper mismatch" for p in problems),
    }
    report["passed"] = all(report["gates"].values())
    return report


# ---------------------------------------------------------------------------
# Mode 2/3: KPI scoring of raw model outputs
# ---------------------------------------------------------------------------
def strict_parse(raw: str):
    """Strict: only surrounding whitespace / a trailing <end_of_turn> is tolerated (no fence stripping)."""
    text = raw.strip()
    if text.endswith(END_TURN):
        text = text[: -len(END_TURN)].strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def score_outputs(evals: list[dict], outputs: list[str]) -> dict:
    assert len(evals) == len(outputs), "one output per eval row required"
    n_valid = 0
    intent_hit = intent_n = 0
    scam_hit = scam_n = 0
    abst_hit = abst_n = 0
    tts_hit = tts_n = 0
    over_abstain = answerable_n = 0
    safe_false_alarm = safe_n = 0
    per_bucket = collections.defaultdict(lambda: [0, 0])
    failures = []

    for row, raw in zip(evals, outputs):
        gold = json.loads(row["completion"])
        pred = strict_parse(raw)
        is_a = row["task_type"] == "assistant"
        validator = validate_task_a if is_a else validate_task_b
        valid = pred is not None and not validator(pred, strict_caps=False)
        n_valid += valid
        ok = False

        if is_a:
            intent_n += 1
            intent_ok = valid and pred["task"] == gold["task"]
            intent_hit += intent_ok
            if valid:
                tts_n += 1
                tts_ok = (devanagari_ok(pred["replyHi"]) and word_count(pred["replyEn"]) <= REPLY_WORD_CAP
                          and word_count(pred["replyHi"]) <= REPLY_WORD_CAP)
                tts_hit += tts_ok
            else:
                tts_n += 1
            if gold["isScamAlert"]:
                scam_n += 1
                s_ok = valid and pred["isScamAlert"] is True and "1930" in pred["replyEn"]
                scam_hit += s_ok
                ok = s_ok
            elif gold["abstained"]:
                abst_n += 1
                a_ok = valid and pred["abstained"] is True and pred["task"] == "NONE"
                abst_hit += a_ok
                ok = a_ok
            else:
                answerable_n += 1
                if valid and pred["abstained"]:
                    over_abstain += 1
                ok = intent_ok and not pred["abstained"]
        else:
            tts_n += 1
            if valid:
                tts_hit += (devanagari_ok(pred["explanationHi"]) and devanagari_ok(pred["actionTipHi"])
                            and all(word_count(pred[k]) <= EXPLANATION_WORD_CAP
                                    for k in ("explanationEn", "explanationHi", "actionTipEn", "actionTipHi")))
            if gold["riskLevel"] == "HIGH_RISK":
                scam_n += 1
                s_ok = valid and pred["riskLevel"] == "HIGH_RISK" and "1930" in pred["actionTipEn"]
                scam_hit += s_ok
                ok = s_ok
            else:
                ok = valid and pred["riskLevel"] == gold["riskLevel"]
                if gold["riskLevel"] == "SAFE":
                    safe_n += 1
                    if valid and pred["riskLevel"] != "SAFE":
                        safe_false_alarm += 1

        per_bucket[row["bucket"]][0] += ok
        per_bucket[row["bucket"]][1] += 1
        if not ok and len(failures) < 25:
            failures.append({"id": row.get("id"), "bucket": row["bucket"], "query": row["query"],
                             "gold": row["completion"], "output": raw[:400]})

    report = {
        "mode": "kpi",
        "eval_rows": len(evals),
        "valid_json_pct": pct(n_valid, len(evals)),
        "intent_accuracy_pct": pct(intent_hit, intent_n),
        "scamguard_recall_pct": pct(scam_hit, scam_n),
        "safe_abstention_pct": pct(abst_hit, abst_n),
        "bilingual_tts_pct": pct(tts_hit, tts_n),
        "over_abstention_pct": pct(over_abstain, answerable_n),
        "scam_safe_false_alarm_pct": pct(safe_false_alarm, safe_n),
        "per_bucket_accuracy_pct": {b: pct(h, n) for b, (h, n) in sorted(per_bucket.items())},
        "failures_sample": failures,
    }
    report["gates"] = {f"{k} >= {v}": report[k] >= v for k, v in KPI_GATES.items()}
    report["passed"] = all(report["gates"].values())
    return report


def generate_outputs(model_dir: str, evals: list[dict], batch_size: int, max_new_tokens: int) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if not torch.cuda.is_available():
        dtype = torch.float32
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"  # renamed in 4.56
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="eager", **{dtype_key: dtype},
    )
    model.eval()
    stop_ids = [tok.eos_token_id, tok.convert_tokens_to_ids(END_TURN)]

    outputs = []
    for i in range(0, len(evals), batch_size):
        batch = evals[i:i + batch_size]
        prompts = [wrap_prompt(r["prompt"]) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 eos_token_id=stop_ids, pad_token_id=tok.pad_token_id)
        for seq in gen[:, enc["input_ids"].shape[1]:]:
            outputs.append(tok.decode(seq, skip_special_tokens=True).strip())
        print(f"  generated {min(i + batch_size, len(evals))}/{len(evals)}", flush=True)
    return outputs


def print_report(report: dict) -> None:
    hidden = {"problems_sample", "failures_sample"}
    print(json.dumps({k: v for k, v in report.items() if k not in hidden}, ensure_ascii=False, indent=2))
    sample = report.get("problems_sample") or report.get("failures_sample")
    if sample:
        print(f"\nFirst {min(5, len(sample))} problems:")
        for p in sample[:5]:
            print("  ", json.dumps(p, ensure_ascii=False)[:500])
    print("\nRESULT:", "PASS" if report["passed"] else "FAIL")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-file")
    ap.add_argument("--eval-file", required=True)
    ap.add_argument("--model-dir")
    ap.add_argument("--predictions-file")
    ap.add_argument("--save-predictions", help="write raw model outputs here (model mode)")
    ap.add_argument("--report-file", help="write the JSON report here")
    ap.add_argument("--limit", type=int, help="score only the first N eval rows")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args(argv)

    evals = read_jsonl(args.eval_file)
    if args.limit:
        evals = evals[: args.limit]

    if args.model_dir or args.predictions_file:
        if args.predictions_file:
            preds = read_jsonl(args.predictions_file)
            by_id = {p["id"]: p["output"] for p in preds if "id" in p}
            outputs = [by_id[r["id"]] for r in evals] if by_id else [p["output"] for p in preds[: len(evals)]]
        else:
            outputs = generate_outputs(args.model_dir, evals, args.batch_size, args.max_new_tokens)
            if args.save_predictions:
                with open(args.save_predictions, "w", encoding="utf-8") as f:
                    for r, o in zip(evals, outputs):
                        f.write(json.dumps({"id": r["id"], "output": o}, ensure_ascii=False) + "\n")
        report = score_outputs(evals, outputs)
    else:
        if not args.train_file:
            ap.error("dataset audit needs --train-file (or pass --model-dir / --predictions-file)")
        report = audit_dataset(read_jsonl(args.train_file), evals)

    print_report(report)
    if args.report_file:
        with open(args.report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

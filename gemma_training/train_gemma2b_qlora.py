#!/usr/bin/env python3
"""QLoRA fine-tune of google/gemma-2-2b-it on a free Colab T4 (guide page 6).

  * 4-bit NF4 + double quant base, attn_implementation="eager" (Gemma-2 softcapping)
  * LoRA r=16, alpha=32, dropout=0.05 on q/k/v/o/gate/up/down projections
  * bs=2 x grad_accum=8 (effective 16), paged_adamw_8bit, gradient checkpointing
  * lr=2e-4 cosine, warmup 8%, 4 epochs, max_seq_len=512 (= LlmInference.setMaxTokens(512))
  * completion-only loss: every token up to and including "<start_of_turn>model\\n" gets label -100,
    so loss is computed on the JSON answer + <end_of_turn> only (same effect as TRL's
    DataCollatorForCompletionOnlyLM, implemented directly so it works across TRL versions)
  * merge_and_unload() -> <output-dir>/merged_fp16 (model.safetensors + tokenizer.model)

  python3 train_gemma2b_qlora.py --model-id google/gemma-2-2b-it \\
      --train-file data/train.jsonl --eval-file data/eval.jsonl \\
      --epochs 4 --output-dir outputs/gemma2-2b-banksathi
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import shutil
import sys

from banksathi_common import END_TURN, MODEL_TURN, read_jsonl

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def dtype_kwarg(dtype) -> dict:
    """transformers >= 4.56 renamed from_pretrained(torch_dtype=...) to dtype=..."""
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="google/gemma-2-2b-it")
    ap.add_argument("--train-file", default="data/train.jsonl")
    ap.add_argument("--eval-file", default="data/eval.jsonl")
    ap.add_argument("--output-dir", default="outputs/gemma2-2b-banksathi")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.08)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-merge", action="store_true", help="skip merge_and_unload (adapter only)")
    ap.add_argument("--merge-only", action="store_true", help="only merge an existing adapter in output-dir/adapter")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Tokenisation with completion-only label masking
# ---------------------------------------------------------------------------
def build_features(rows, tok, max_len):
    feats, dropped = [], 0
    for r in rows:
        text = r["text"]
        cut = text.index(MODEL_TURN) + len(MODEL_TURN)  # template boundary
        prompt_ids = tok(text[:cut], add_special_tokens=True)["input_ids"]  # adds <bos>
        completion_ids = tok(text[cut:], add_special_tokens=False)["input_ids"]
        ids = prompt_ids + completion_ids
        if len(ids) > max_len:  # never truncate the JSON target - drop instead
            dropped += 1
            continue
        feats.append({
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prompt_ids) + completion_ids,
        })
    return feats, dropped


class CompletionOnlyCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch

        width = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            pad = width - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [self.pad_id] * pad)
            out["attention_mask"].append(b["attention_mask"] + [0] * pad)
            out["labels"].append(b["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def sanity_check_masking(feats, tok):
    f = feats[0]
    trained = [t for t, l in zip(f["input_ids"], f["labels"]) if l != -100]
    decoded = tok.decode(trained)
    assert decoded.lstrip().startswith("{"), f"loss span should start at JSON, got: {decoded[:80]!r}"
    assert END_TURN in decoded, "loss span must include <end_of_turn> so the model learns to stop"
    print(f"[mask check] trained span: {decoded[:120]}...")


def training_args(args, fp16: bool, bf16: bool, n_train: int):
    import math

    from transformers import TrainingArguments

    steps_per_epoch = math.ceil(n_train / (args.batch_size * args.grad_accum))
    total_steps = max(1, math.ceil(steps_per_epoch * args.epochs))
    kwargs = dict(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, round(total_steps * args.warmup_ratio)),  # 8% warmup (warmup_ratio removed in transformers 5)
        optim="paged_adamw_8bit",
        weight_decay=0.0,
        max_grad_norm=0.3,
        fp16=fp16,
        bf16=bf16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )
    # transformers renamed evaluation_strategy -> eval_strategy in 4.41
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "epoch"
    dropped = sorted(k for k in kwargs if k not in params)
    if dropped:
        print(f"note: this transformers version does not accept {dropped}; skipping them")
    print(f"schedule: {total_steps} optimizer steps, warmup {kwargs['warmup_steps']}")
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in params})


def train(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, set_seed

    set_seed(args.seed)
    if not torch.cuda.is_available():
        sys.exit("A CUDA GPU is required (Colab: Runtime -> Change runtime type -> T4 GPU).")
    bf16 = torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8
    compute_dtype = torch.bfloat16 if bf16 else torch.float16  # T4 (sm_75) -> fp16
    print(f"GPU: {torch.cuda.get_device_name(0)} | compute dtype: {compute_dtype}")

    tok = AutoTokenizer.from_pretrained(args.model_id)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_rows, eval_rows = read_jsonl(args.train_file), read_jsonl(args.eval_file)
    train_feats, d1 = build_features(train_rows, tok, args.max_seq_len)
    eval_feats, d2 = build_features(eval_rows, tok, args.max_seq_len)
    lengths = [len(f["input_ids"]) for f in train_feats]
    print(f"train {len(train_feats)} (dropped {d1} > {args.max_seq_len} tok) | eval {len(eval_feats)} (dropped {d2})"
          f" | max len {max(lengths)} | mean len {sum(lengths) / len(lengths):.0f}")
    sanity_check_masking(train_feats, tok)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb,
        device_map={"": 0},
        attn_implementation="eager",  # mandatory for Gemma-2 logit softcapping during training
        **dtype_kwarg(compute_dtype),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=training_args(args, fp16=not bf16, bf16=bf16, n_train=len(train_feats)),
        train_dataset=Dataset.from_list(train_feats),
        eval_dataset=Dataset.from_list(eval_feats),
        data_collator=CompletionOnlyCollator(tok.pad_token_id),
    )
    result = trainer.train()
    metrics = trainer.evaluate()
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"peak VRAM allocated: {peak_gb:.2f} GB | eval_loss: {metrics.get('eval_loss')}")

    adapter_dir = os.path.join(args.output_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    with open(os.path.join(args.output_dir, "train_metrics.json"), "w") as f:
        json.dump({"train": result.metrics, "eval": metrics, "peak_vram_gb": round(peak_gb, 2),
                   "train_rows": len(train_feats), "eval_rows": len(eval_feats),
                   "dropped_overlength": d1 + d2, "args": vars(args)}, f, indent=2)
    print(f"adapter saved -> {adapter_dir}")

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


def merge(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = os.path.join(args.output_dir, "adapter")
    merged_dir = os.path.join(args.output_dir, "merged_fp16")
    print(f"merging {adapter_dir} into fp16 base ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id, attn_implementation="eager",
        device_map="auto" if torch.cuda.is_available() else None, **dtype_kwarg(torch.float16),
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.generation_config.do_sample = False
    merged.save_pretrained(merged_dir, max_shard_size="6GB")  # safetensors by default
    tok = AutoTokenizer.from_pretrained(args.model_id)
    tok.save_pretrained(merged_dir)

    # MediaPipe bundler needs the SentencePiece tokenizer.model next to the weights
    sp_path = os.path.join(merged_dir, "tokenizer.model")
    local_sp = os.path.join(args.model_id, "tokenizer.model")
    if not os.path.exists(sp_path) and os.path.exists(local_sp):  # local base dir (e.g. Kaggle Models)
        shutil.copy(local_sp, sp_path)
    if not os.path.exists(sp_path):
        from huggingface_hub import hf_hub_download
        shutil.copy(hf_hub_download(args.model_id, "tokenizer.model"), sp_path)
    print(f"merged fp16 model saved -> {merged_dir}")
    print("files:", sorted(os.listdir(merged_dir)))


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.merge_only:
        train(args)
    if not args.no_merge:
        merge(args)


if __name__ == "__main__":
    main()

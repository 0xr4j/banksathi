# BankSathi · Gemma-2B QLoRA training pipeline

The code for the *Gemma-2B QLoRA Training & 1k-3k Dataset Engineering Guide*. It fine-tunes
`google/gemma-2-2b-it` on a free Colab T4 so it fits the two `HybridAiService.kt` JSON contracts. Then it
converts the result to an INT4 MediaPipe `.task` bundle.

| File | Guide stage | What it does |
|---|---|---|
| `banksathi_common.py` | shared | Prompt headers, JSON schemas, Gemma-2 chat wrapper, SHA-256 dedupe, Devanagari check |
| `dataset_generator.py` | Stage 2 | Stratified 1k-3k dataset, 3 registers + ASR noise, RAG grounding, abstention traps, ScamGuard tiers |
| `eval_gemma_banksathi.py` | Stage 4 | Pre-GPU dataset audit **and** live 5-KPI hold-out gate (or scores saved outputs) |
| `train_gemma2b_qlora.py` | Stage 3 | 4-bit NF4 QLoRA, completion-only loss masking, `merge_and_unload()` to `merged_fp16/` |
| `convert_to_mediapipe.py` | Stage 4 | ai-edge-torch INT4 TFLite, MediaPipe bundler, `banksathi-cpu-int4.task` |
| `BankSathi_Gemma2B_Colab.ipynb` | all | Click-through Colab runbook |

## Run it on Colab

1. Upload `BankSathi_Gemma2B_Colab.ipynb` to https://colab.research.google.com and choose a **T4 GPU** runtime.
2. Run the cells top to bottom. When asked, upload `gemma_training.zip`, which sits next to this folder.
3. Paste a Hugging Face token. You must accept the `google/gemma-2-2b-it` licence first.

You can also run it from a terminal (guide p.7):

```bash
pip install -q -r requirements.txt && huggingface-cli login
python3 dataset_generator.py --out-dir data --target-rows 1500
python3 eval_gemma_banksathi.py --train-file data/train.jsonl --eval-file data/eval.jsonl   # must PASS
python3 train_gemma2b_qlora.py --model-id google/gemma-2-2b-it --train-file data/train.jsonl \
    --eval-file data/eval.jsonl --epochs 4 --output-dir outputs/gemma2-2b-banksathi
python3 eval_gemma_banksathi.py --model-dir outputs/gemma2-2b-banksathi/merged_fp16 --eval-file data/eval.jsonl
# fresh runtime:
pip install -r requirements-convert.txt
python3 convert_to_mediapipe.py --merged-dir outputs/gemma2-2b-banksathi/merged_fp16 \
    --context-len 512 --output-task outputs/gemma2-2b-banksathi-cpu-int4.task
```

## Notes from the first full run (Colab, Nov 2026)

* Colab can hand out a CPU-only runtime even when T4 is selected. Check that `nvidia-smi` shows `Tesla T4`.
* The training script now handles `transformers` 5.x: it computes `warmup_steps` and passes `dtype=`.
* Colab's preinstalled `torchao` breaks the PEFT merge, so the notebook uninstalls it.
* `convert_to_mediapipe.py` needs **at least 25-30 GB of system RAM**. The free Colab machine (12 GB) is
  OOM-killed while it builds the graph, and swap is not allowed on Colab. Run it on Colab High-RAM, Kaggle
  (30 GB) or a PC with 32 GB or more. Pin `mediapipe==0.10.35`, because versions 1.x dropped the
  `genai` bundler.
* First run: 4 of 5 gates passed. JSON was 98.67%: 3 Hinglish KCC rows answered `"task":"LOAN"`. To fix it,
  add more Hinglish KCC templates and/or train for `--epochs 5`.

## Dataset design (guide pp. 3-5)

* **Quotas:** the weights are loan_kcc .14, passbook_dbt_bsbda .14, transfer_ippb_bc .14, safe_abstention .16,
  withdraw_bc_le_10k .12, withdraw_branch_gt_10k .10, assistant_scam_alert .10, scam_high_risk .04,
  scam_caution .03 and scam_safe .03. You can change them in `BUCKET_WEIGHTS`.
* **Registers:** about 38% English, 24% Devanagari Hindi and 38% Hinglish. The Hinglish rows get simulated
  Whisper-tiny slips: lowercase text, a dropped `?`, `aadhar`, `pasbuk`, `rupees`, fillers and stutters. The
  targets always stay as clean English plus Devanagari.
* **Boundaries:** amounts of Rs 9,999, 10,000 and 10,001 test BC versus Branch/ATM routing. For KCC, amounts of
  Rs 1,99,999, 2,00,000, 2,00,001 and 2,50,000 test the collateral-free limit. Withdrawals at or below Rs 10k and
  above Rs 10k use the *same* query templates, so the model learns the threshold rather than the wording.
* **Grounding:** about 50% of Task A prompts carry a `[SOURCE_ID | Verified 2026-11-14]` snippet. 30% of those
  also carry a distractor snippet. Abstention traps can come with an on-topic snippet that still does not
  answer the question.
* **Gates:** queries are deduplicated by SHA-256 after removing punctuation and case, and the split is a
  stratified 85/15. Leakage is checked and must be 0. Every target is checked against the schema, the
  45-word limit (50 for Task B) and Devanagari parity.

Each JSONL row contains `text`, which is the Gemma-2 chat turn used for training. It also contains
`prompt`, `completion`, `bucket`, `register`, `grounded`, `context_ids`, `query` and `query_sha256`.

## Prompt contract

The prompt headers in `banksathi_common.py` (`TASK_A_HEADER`, `TASK_B_HEADER`, `build_task_a_prompt`) must match
what `HybridAiService.kt` sends **exactly**: the same header, the `Verified Context:` block, then `User: <query>`.
If the app's wording differs, change it in that one file and regenerate the dataset. Otherwise the model is
served prompts it was never trained on. The app should wrap prompts as
`<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n`, unless you bundle with
`--embed-chat-template`.

## Failure playbook (guide p.7)

| Symptom | Fix |
|---|---|
| `valid_json_pct` < 99 | Check that the `[mask check]` line printed during training starts at `{`. Try `--epochs 5`. |
| WITHDRAW predicted for transfer/DBT queries | Keep withdraw_* at or below 25% (the audit enforces this). Raise the transfer/passbook weights. |
| `scamguard_recall_pct` < 100 | Add the missed phrasing to `SCAM_ALERT_Q` / `HIGH_RISK_MSGS`. Keep the Room severity rules. |
| `safe_abstention_pct` < 95 | Raise `safe_abstention` from 0.16 to 0.22. Add more templates to `ABSTAIN_Q`. |
| `replyHi` loses Devanagari or runs past 45 words | Shorten the reply templates. The audit's `devanagari_ok` will flag it. |

Failed rows are printed under `failures_sample`, and `--save-predictions` keeps every raw output.

## Android smoke test (airplane mode, guide p.8)

1. `Bhaiya 5000 rupaye nikalne hain` → WITHDRAW, 5000, Bank Mitra, no fee
2. `25000 rupees nikalne hain` → WITHDRAW, 25000, Branch/ATM
3. `150000 KCC loan documents?` → LOAN_KCC, Rs 2L collateral-free, Khatauni + 2 copies + 3 photos
4. `Caller asked OTP & AnyDesk` → ScamGuard fires before the LLM, isScamAlert=true, 1930 + 3 days
5. `Exact balance in SBI acct 30491?` → abstained=true, task NONE, Branch / 14448
6. A PM-KISAN credit SMS → riskLevel SAFE

Keep `BankingRuleEngine.isImmediateScamPhrase()` running before inference. Keep the deterministic fallback
for when JSON parsing fails. Fine-tuning does not replace either one.

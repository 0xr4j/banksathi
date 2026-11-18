# BankSathi

Offline, bilingual (English + Hindi) banking assistant for rural India. This repo holds the on-device
LLM pipeline: a `google/gemma-2-2b-it` model fine-tuned with QLoRA so it answers in the two JSON
contracts used by the Android app's `HybridAiService.kt`, then converted to an INT4 MediaPipe `.task`
bundle that runs fully offline on the phone.

## What's inside

| Path | What it does |
|---|---|
| `gemma_training/dataset_generator.py` | Builds a stratified 1k-3k row dataset: English / Hindi / Hinglish, ASR noise, RBI-grounded context, abstention traps, ScamGuard tiers |
| `gemma_training/eval_gemma_banksathi.py` | Pre-GPU dataset audit and the 5-KPI hold-out safety gate |
| `gemma_training/train_gemma2b_qlora.py` | 4-bit NF4 QLoRA on a free Colab T4, completion-only loss, merge to FP16 |
| `gemma_training/convert_to_mediapipe.py` | INT4 TFLite conversion and MediaPipe `.task` bundling |
| `gemma_training/BankSathi_Gemma2B_Colab*.ipynb` | Click-through Colab runbooks |

See [`gemma_training/README.md`](gemma_training/README.md) for how to run it, the dataset design and the
failure playbook.

## First training run

4 of 5 KPI gates passed on the 226-row hold-out set: ScamGuard recall 100%, safe abstention 100%,
intent accuracy and bilingual TTS above target. Valid JSON was 98.67% against a 99% target.

## Credits

* **Raj Jha** ([@0xr4j](https://github.com/0xr4j))
* **Nikhil**

Built together by Raj and Nikhil.

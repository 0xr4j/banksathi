"""Merge the BankSathi LoRA into Gemma-2 2B and export both deployment formats on a free Kaggle CPU notebook.

This is the exact recipe that produced the published files after free Colab (12 GB RAM) and even Kaggle
(31 GB RAM) ran out of memory with the litert-torch INT4 exporter.

Kaggle setup (no Hugging Face token needed):
  * Input dataset with adapter_model.safetensors (or its .partNN chunks), adapter_config.json and
    train_gemma2b_qlora.py from this repo.
  * Input model: google/gemma-2 -> Transformers -> gemma-2-2b-it (accept the Gemma licence on Kaggle first).
  * Settings: Accelerator None, Internet on.

Outputs in /kaggle/working:
  * gemma2-2b-banksathi-cpu-int4.bin  MediaPipe LLM Inference, CPU, int4 FFN / int8 attention (~1.67 GB, ~3 GB RAM)
  * gguf/banksathi-gemma2-2b-Q8_0-0000N-of-00006.gguf  llama.cpp / Ollama / browser demo (~2.78 GB)

Run as a notebook cell or `python kaggle_merge_and_convert.py [--skip-mediapipe] [--skip-gguf]`.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

WORK = "/kaggle/working"
OUT = "/tmp/out"
MERGED = f"{OUT}/merged_fp16"
LLAMA_CPP_TARBALL = "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/heads/master"


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def find_inputs() -> tuple[str, str]:
    adapter_dir = os.path.dirname(glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)[0])
    base_dir = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/tokenizer.model", recursive=True)
                if "banksathi" not in p][0]
    return adapter_dir, base_dir


def merge(adapter_dir: str, base_dir: str) -> None:
    os.makedirs(f"{OUT}/adapter", exist_ok=True)
    parts = sorted(glob.glob(f"{adapter_dir}/adapter_model.safetensors.part*"))
    src = " ".join(parts) if parts else f"{adapter_dir}/adapter_model.safetensors"
    sh(f"cat {src} > {OUT}/adapter/adapter_model.safetensors")
    sh(f"cp {adapter_dir}/adapter_config.json {OUT}/adapter/")
    # Kaggle's preinstalled torchao breaks the PEFT merge; litert-torch needs it, the steps below do not.
    sh("pip uninstall -y -q torchao; pip install -q peft")
    # train_gemma2b_qlora.py imports banksathi_common only for training; a stub is enough for --merge-only.
    with open("/tmp/banksathi_common.py", "w") as f:
        f.write("END_TURN = '<end_of_turn>'\nMODEL_TURN = '<start_of_turn>model\\n'\nread_jsonl = None\n")
    sh(f"cd /tmp && PYTHONPATH=/tmp python {adapter_dir}/train_gemma2b_qlora.py --merge-only "
       f"--model-id {base_dir} --output-dir {OUT}")


def to_mediapipe() -> None:
    sh("pip install -q mediapipe==0.10.35")
    from mediapipe.tasks.python.genai import converter

    cfg = converter.ConversionConfig(
        input_ckpt=MERGED + "/", ckpt_format="safetensors", model_type="GEMMA2_2B", backend="cpu",
        output_dir="/tmp/mp_work", combine_file_only=False, vocab_model_file=f"{MERGED}/tokenizer.model",
        output_tflite_file=f"{WORK}/gemma2-2b-banksathi-cpu-int4.bin", is_quantized=True,
        attention_quant_bits=8, feedforward_quant_bits=4, embedding_quant_bits=8,
    )
    converter.convert_checkpoint(cfg)
    sh("rm -rf /tmp/mp_work")


def to_gguf() -> None:
    # Pure Python conversion: no compiler needed. Q8_0, split into ~512 MB shards for the browser (2 GB limit).
    sh(f"cd /tmp && curl -sL {LLAMA_CPP_TARBALL} | tar xz")
    sh("pip install -q /tmp/llama.cpp-master/gguf-py")
    os.makedirs(f"{WORK}/gguf", exist_ok=True)
    sh(f"cd /tmp/llama.cpp-master && python convert_hf_to_gguf.py {MERGED} --outtype q8_0 --split-max-size 512M "
       f"--outfile {WORK}/gguf/banksathi-gemma2-2b-Q8_0.gguf")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-mediapipe", action="store_true")
    ap.add_argument("--skip-gguf", action="store_true")
    args = ap.parse_args(argv)
    adapter_dir, base_dir = find_inputs()
    print("adapter:", adapter_dir, "| base:", base_dir)
    merge(adapter_dir, base_dir)
    if not args.skip_mediapipe:
        to_mediapipe()
    if not args.skip_gguf:
        to_gguf()
    sh(f"ls -laR {WORK}")


if __name__ == "__main__":
    main(sys.argv[1:] if "ipykernel" not in sys.modules else [])

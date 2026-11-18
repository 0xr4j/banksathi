#!/usr/bin/env python3
"""merged_fp16 -> INT4 TFLite (ai-edge-torch) -> MediaPipe .task bundle (guide page 8).

  python3 convert_to_mediapipe.py \\
      --merged-dir outputs/gemma2-2b-banksathi/merged_fp16 \\
      --context-len 512 --output-task outputs/gemma2-2b-banksathi-cpu-int4.task

Run in a FRESH Colab runtime after `pip install -r requirements-convert.txt`; ai-edge-torch pins its own
torch version that conflicts with the training stack. Needs ~12 GB system RAM (Colab High-RAM helps).

The ai-edge-torch generative API changed between releases, so signatures are detected at runtime.
"""
from __future__ import annotations

import argparse
import glob
import inspect
import os
import sys
import tempfile


def import_generative():
    for prefix in ("ai_edge_torch.generative", "litert_torch.generative"):
        try:
            gemma2 = __import__(f"{prefix}.examples.gemma", fromlist=["gemma2"]).gemma2
            converter = __import__(f"{prefix}.utilities", fromlist=["converter"]).converter
            return gemma2, converter
        except (ImportError, AttributeError):
            continue
    sys.exit("ai-edge-torch is not installed: pip install -r requirements-convert.txt")


def build_model(gemma2, merged_dir: str, context_len: int):
    params = inspect.signature(gemma2.build_2b_model).parameters
    kwargs = {"kv_cache_max_len": context_len} if "kv_cache_max_len" in params else {}
    if "mask_cache_size" in params:  # litert-torch >= 0.10: causal mask cache must match the KV cache
        kwargs["mask_cache_size"] = context_len
    return gemma2.build_2b_model(merged_dir, **kwargs)


def to_tflite(converter, model, work_dir: str, prefill: int, quantize: str) -> str:
    fn = converter.convert_to_tflite
    params = inspect.signature(fn).parameters
    if "output_name_prefix" in params:  # ai-edge-torch >= 0.3
        extra = {"kv_cache_max_len": prefill} if "kv_cache_max_len" in params else {}  # required in litert-torch >= 0.10
        if "export_config" in params and hasattr(converter, "export_config_lib"):
            extra["export_config"] = converter.export_config_lib.ExportConfig()  # None crashes litert-torch 0.10
        fn(model, output_path=work_dir, output_name_prefix="gemma2_2b_banksathi",
           prefill_seq_len=prefill, quantize=quantize, **extra)
    else:  # 0.2.x: tflite_path + boolean quantize (dynamic int8)
        print("note: old ai-edge-torch detected; it only supports dynamic INT8, not INT4")
        fn(model, os.path.join(work_dir, "gemma2_2b_banksathi.tflite"), prefill_seq_len=prefill, quantize=True)
    found = sorted(glob.glob(os.path.join(work_dir, "*.tflite")), key=os.path.getsize, reverse=True)
    if not found:
        sys.exit("conversion produced no .tflite file")
    return found[0]


def bundle(tflite_path: str, tokenizer_path: str, output_task: str, embed_template: bool):
    from mediapipe.tasks.python.genai import bundler

    fields = inspect.signature(bundler.BundleConfig).parameters
    cfg = dict(
        tflite_model=tflite_path,
        tokenizer_model=tokenizer_path,
        start_token="<bos>",
        stop_tokens=["<eos>", "<end_of_turn>"],
        output_filename=output_task,
        enable_bytes_to_unicode_mapping=False,
    )
    if embed_template and "prompt_prefix" in fields:
        # Only if HybridAiService.kt sends the raw prompt WITHOUT Gemma turn markers.
        cfg["prompt_prefix"] = "<start_of_turn>user\n"
        cfg["prompt_suffix"] = "<end_of_turn>\n<start_of_turn>model\n"
    bundler.create_bundle(bundler.BundleConfig(**cfg))


def mediapipe_convert(merged_dir: str, tokenizer_path: str, output_bin: str):
    """Low-RAM path (~3 GB): MediaPipe GenAI converter quantizes layer by layer straight from safetensors.

    Output is a CPU .bin (int4 FFN, int8 attention/embedding) that LlmInference loads directly.
    """
    from mediapipe.tasks.python.genai import converter

    work = tempfile.mkdtemp(prefix="mp_work_")
    cfg = converter.ConversionConfig(
        input_ckpt=merged_dir.rstrip("/") + "/", ckpt_format="safetensors", model_type="GEMMA2_2B",
        backend="cpu", output_dir=work, combine_file_only=False, vocab_model_file=tokenizer_path,
        output_tflite_file=output_bin, is_quantized=True,
        attention_quant_bits=8, feedforward_quant_bits=4, embedding_quant_bits=8,
    )
    converter.convert_checkpoint(cfg)
    import shutil
    shutil.rmtree(work, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged-dir", required=True)
    ap.add_argument("--engine", choices=["mediapipe", "litert"], default="mediapipe",
                    help="mediapipe: low-RAM GenAI converter -> .bin (default, works on 12 GB Colab); "
                         "litert: litert-torch INT4 .task (needs >31 GB RAM)")
    ap.add_argument("--context-len", type=int, default=512, help="prefill + KV cache length")
    ap.add_argument("--output-task", default="outputs/gemma2-2b-banksathi-cpu-int4.task")
    ap.add_argument("--quantize", default="dynamic_int4_block32",
                    help="ai-edge-torch recipe: dynamic_int4_block32 | dynamic_int4_block128 | dynamic_int8")
    ap.add_argument("--embed-chat-template", action="store_true",
                    help="store Gemma turn markers in the bundle (use only if the app does not add them)")
    ap.add_argument("--keep-tflite", help="also copy the intermediate .tflite here")
    args = ap.parse_args(argv)

    tokenizer_path = os.path.join(args.merged_dir, "tokenizer.model")
    if not os.path.exists(tokenizer_path):
        sys.exit(f"missing {tokenizer_path} (re-run train_gemma2b_qlora.py --merge-only)")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_task)), exist_ok=True)

    if args.engine == "mediapipe":
        out = os.path.splitext(args.output_task)[0] + ".bin"
        print("MediaPipe GenAI converter (GEMMA2_2B, cpu, int4 FFN / int8 attn) ->", out)
        mediapipe_convert(args.merged_dir, tokenizer_path, out)
        print(f"done -> {out} ({os.path.getsize(out) / 1024 ** 3:.2f} GB)")
        return

    gemma2, converter = import_generative()
    print("building ai-edge-torch Gemma-2 2B graph from", args.merged_dir)
    model = build_model(gemma2, args.merged_dir, args.context_len)
    with tempfile.TemporaryDirectory() as work:
        print(f"converting -> TFLite ({args.quantize}, prefill={args.context_len}, kv={args.context_len}) ...")
        tflite = to_tflite(converter, model, work, args.context_len, args.quantize)
        print(f"tflite: {os.path.getsize(tflite) / 1024 ** 3:.2f} GB")
        if args.keep_tflite:
            import shutil
            shutil.copy(tflite, args.keep_tflite)
        print("bundling with MediaPipe ...")
        bundle(tflite, tokenizer_path, args.output_task, args.embed_chat_template)
    print(f"done -> {args.output_task} ({os.path.getsize(args.output_task) / 1024 ** 3:.2f} GB)")
    print("adb push", args.output_task, "/data/local/tmp/llm/gemma2-2b-it-cpu-int4.task")


if __name__ == "__main__":
    main()

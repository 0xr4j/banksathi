---
title: BankSathi
emoji: 🏦
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: gemma
short_description: Offline bilingual banking assistant, Gemma-2 2B in-browser
models:
  - 0xr4j/banksathi-gemma2-2b
custom_headers:
  cross-origin-embedder-policy: credentialless
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# BankSathi demo

[BankSathi Gemma-2 2B](https://huggingface.co/0xr4j/banksathi-gemma2-2b) running fully in your browser with
[wllama](https://github.com/ngxson/wllama) (llama.cpp compiled to WebAssembly). Nothing you type leaves your
device. The Q8_0 GGUF model (~2.8 GB) downloads once and is cached. Works best in desktop Chrome or Edge with
8 GB+ RAM.

Not financial advice. Always confirm with your bank.

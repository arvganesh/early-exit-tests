# Self-Speculation Experiments

This repo contains the code used in my investigation of "self-speculation", a variant of [speculative decoding](https://research.google/blog/looking-back-at-speculative-decoding/).

## Environment

- Using the following [NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags): `nvcr.io/nvidia/pytorch:25.01-py3`
- GPU: NVIDIA GH200 120 GB

## Training Performance Notes

- `train.py` now supports `--attn_implementation {auto,sdpa,flash_attention_2}`, `--torch_dtype {auto,bf16,fp16,fp32}`, `--tf32/--no_tf32`, and optional `--torch_compile`.
- To run in the background with a log file (tmux if available, else nohup): `./launch_train_bg.sh <session_name> python train.py ...`

## Repository File Structure

This repository contains various scripts for training and evaluating models, handling datasets, and utilities for working with LLaMA-based models.

### 📂 File Overview

#### 🔹 Core Scripts
- **`train.py`** – Main script for training models.
- **`evaluate.py`** – Script for evaluating trained models.

#### 🔹 Dataset Handling
- **`share_gpt_dataset.py`** – Data loading for ShareGPT dataset.
- **`slim_pj_dataset.py`** – Data loading for the SlimPajama dataset.
- **`xsum_dataset.py`** – Data loading  for the the XSum dataset.

#### 🔹 LLaMA Model Variants
- **`truncated_llama.py`** – Implements a truncated version of the LLaMA model.
- **`truncated_llama_test.py`** – Test script for `truncated_llama.py`.
- **`tuned_llama.py`** – A fine-tuned version of the LLaMA model.

#### 🔹 Utilities
- **`data_utils.py`** – Contains helper functions for data processing.

#### 🔹 Miscellaneous
- **`README.md`** – Documentation for the repository.
- **`benchmark_torch_compile.py`** – Benchmarking PyTorch compilation performance as applied to HuggingFace LLMs and early-exiting variants.

## Findings
Evaluated on UltraChat prompts with chat-style decoding (temp=0.7, top_p=0.95):

- Training only the exit head (200 steps) produced near-zero acceptance rates at early layers (layers 3/7) and weak results even at layer 11 (~0.12%).
- Adding capacity by fine-tuning the last transformer block significantly improved results, jumping layer 11 to ~1.61% acceptance rate with much better perplexity.
- Learning rate and schedule matter: the best layer-11 run used LR=5e-5 with a constant schedule, achieving ~1.49% acceptance rate and ~20.8 perplexity on UltraChat.
- Even the best configurations averaged ~0.12 accepted draft tokens per step (gamma=8), which is too low for wall-clock speedup. This suggests the bottleneck is model capacity at the exit point rather than train/eval distribution mismatch.

Next steps (not completed): Two-stage training with longer head-only warmup followed by fine-tuning the last transformer block, and experimenting with deeper exit layers (layer 13) to see if acceptance improves.


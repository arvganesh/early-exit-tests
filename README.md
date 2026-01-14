# Self-Speculation Experiments

This repo contains the code used in my investigation of "self-speculation", a variant of [speculative decoding](https://research.google/blog/looking-back-at-speculative-decoding/).

## Environment

- Using the following [NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags): `nvcr.io/nvidia/pytorch:25.01-py3`
- GPU: NVIDIA GH200 120 GB

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
- **`scratch_work.py`** – Temporary or experimental code snippets.

## Findings
coming soon!

####
Clarifiying Questions:
  - Is the “another model” teacher always the same base model at full depth (self-distill), or do you want
    teacher_model_path to be different (bigger/different checkpoint)?
  - Do you intend to fine-tune only the moved head, or also add a learned “exit norm” / small adapter before
    the head? (Right now you’re effectively applying the final RMSNorm at the early layer, frozen.)
  - For speculative decoding, are you targeting HF generate-style KV-cached decoding, or are you OK with a
    custom cached loop?
  - What sampling regime matters for your experiments (temp=0 greedy vs temp>0 multinomial), and do you care
    about matching the exact sampled distribution or just improving accept rate?
  - What’s your main metric: wall-clock speed on GH200, acceptance rate vs layer, perplexity vs layer, or
    some quality metric on downstream prompts?

Metrics To Report:

A practical, report-friendly grid (small but meaningful):
  - Fix top_p=0.95, top_k=0
  - Temperatures: T ∈ {0.7, 1.0} (or {0.8, 1.0})
  - Gammas: γ ∈ {2, 4, 8, 16} (often you’ll find 4–8 best)

What to report:
  - Core metric plots vs exit layer at one policy + one γ:
      - avg_accepted (accepted draft tokens per target call)
      - avg_overlap or avg_tv (distribution closeness; policy-dependent)
  - Ablation: tokens/sec (or avg_accepted) vs γ for 2–3 representative exit layers (early/mid/late), at the
    same T/top_p.
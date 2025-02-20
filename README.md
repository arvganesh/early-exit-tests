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

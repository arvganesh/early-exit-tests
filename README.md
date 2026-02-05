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
- **`scratch_work.py`** – Temporary or experimental code snippets.

## Findings
1/24 Log:

• Here’s where we are (chat + production-like eval = temp=0.7, top_p=0.95, UltraChat prompts), and what it implies.

  - Evals run (commands saved as cmd.txt in each output dir):
      - evaluations/chat_prod_sweep_ultrachat_20260120_172416_head_temp0.7_top_p0.95_20260125_003902/results.csv:1
      - evaluations/chat_prod_sweep_ultrachat_20260120_172416_ftlast_temp0.7_top_p0.95_20260125_003902/results.csv:1
      - evaluations/chat_prod_layer_sweep_ultrachat_fthead_lr5e-5_steps600_20260120_223619_seq_temp0.7_top_p0.95_20260125_005544/results.csv:1
      - evaluations/chat_prod_lr_sweep_ultrachat_layer11_20260120_195052_temp0.7_top_p0.95_20260125_005544/results.csv:1
  - Key takeaways:
      - Head-only @ 200 steps is basically dead for early exits (layers 3/7: spec_accept_rate=0), and weak even at layer 11 (~0.12%).
      - Capacity helps a lot: ft_last_transformer jumps layer 11 to ~1.61% accept rate and much better ppl.
      - Optimization also matters: the best layer-11 LR run is begin1768941207 (this corresponds to the 4th run in run_lr_sweep_ultrachat_layer11.sh, i.e. LR=5e-5 + lr_schedule=constant)
        with spec_accept_rate~1.49%, ultrachat_ppl~20.8.
      - Even the best runs are still at spec_mean_accepted≈0.12 with gamma=8 → acceptance is still far too low for speedup>1, so we’re mostly optimization/capacity-limited, not data-
        mismatch-limited (we deferred mismatch tests).

  Next training run I’d do

  - Two-stage on layer 11 (and add layer 13 to see if deeper exits unlock much higher acceptance):
      - Stage 1 (head-only): constant LR 5e-5, longer run (e.g. 5k–20k steps).
      - Stage 2 (head + last block): init from stage-1 checkpoint, LR smaller (e.g. 2e-5), similar steps.
  - Use existing driver: early-exit-tests/run_sweep_ultrachat_two_stage.sh:1 with env overrides, e.g.
    LAYERS="11 13" MAX_STEPS_HEAD=5000 LR_HEAD=5e-5 MAX_STEPS_FTLAST=5000 LR_FTLAST=2e-5 SPEC_T=0.7 SPEC_TOP_P=0.95 ./run_sweep_ultrachat_two_stage.sh


import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import (
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset
from data_utils import custom_collate_fn
import wandb
import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
import logging
import pickle
import math
import time
import argparse
import sys
from datetime import datetime
import contextlib

from truncated_llama import TruncatedLlama
from share_gpt_dataset import get_sharegpt_dataloaders
from fineweb_dataset import get_fineweb_dataloaders
from ultrachat_dataset import get_ultrachat_dataloaders
from data_utils import get_toy_dataloaders
parser = argparse.ArgumentParser(description="Train a truncated Llama model with a tuned head.")
parser.add_argument(
    "--model_path",
    type=str,
    default="meta-llama/Llama-2-7b-chat-hf",
    help="Path to the model checkpoint or model identifier."
)
parser.add_argument(
    "--run_type",
    type=str,
    help="Path to the model checkpoint or model identifier."
)
parser.add_argument(
    "--target_layer",
    type=int,
    default=16,
    help="The transformer layer number that you want to target."
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Seed for dataset split."
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["fineweb", "sharegpt", "ultrachat", "toy"],
    default="fineweb",
    help="Dataset to use for training/eval.",
)
parser.add_argument(
    "--sharegpt_filter_non_english",
    action="store_true",
    default=True,
    help="Filter ShareGPT to English-only using langdetect (slow; default on).",
)
parser.add_argument(
    "--no_sharegpt_filter_non_english",
    action="store_false",
    dest="sharegpt_filter_non_english",
    help="Disable ShareGPT language filtering (faster).",
)
parser.add_argument(
    "--fineweb_streaming",
    action="store_true",
    default=True,
    help="Use streaming FineWeb loading (recommended).",
)
parser.add_argument(
    "--no_fineweb_streaming",
    action="store_false",
    dest="fineweb_streaming",
    help="Disable streaming for FineWeb (not recommended; slow for large splits).",
)
parser.add_argument(
    "--fineweb_shuffle_buffer_size",
    type=int,
    default=10_000,
    help="Shuffle buffer size for streaming FineWeb.",
)
parser.add_argument(
    "--fineweb_train_examples",
    type=int,
    default=None,
    help="Optional cap on number of FineWeb training examples (dry runs).",
)
parser.add_argument(
    "--fineweb_val_examples",
    type=int,
    default=512,
    help="Number of FineWeb validation examples.",
)
parser.add_argument(
    "--fineweb_test_examples",
    type=int,
    default=512,
    help="Number of FineWeb test examples.",
)
parser.add_argument(
    "--ultrachat_streaming",
    action="store_true",
    default=True,
    help="Use streaming UltraChat loading (recommended).",
)
parser.add_argument(
    "--no_ultrachat_streaming",
    action="store_false",
    dest="ultrachat_streaming",
    help="Disable streaming for UltraChat (not recommended; slow for full splits).",
)
parser.add_argument(
    "--ultrachat_shuffle_buffer_size",
    type=int,
    default=10_000,
    help="Shuffle buffer size for streaming UltraChat.",
)
parser.add_argument(
    "--ultrachat_train_examples",
    type=int,
    default=None,
    help="Optional cap on number of UltraChat training examples (dry runs).",
)
parser.add_argument(
    "--ultrachat_val_examples",
    type=int,
    default=2048,
    help="Number of UltraChat validation examples (from test_sft).",
)
parser.add_argument(
    "--ultrachat_test_examples",
    type=int,
    default=2048,
    help="Number of UltraChat test examples (from test_sft).",
)
parser.add_argument(
    "--ultrachat_date_string",
    type=str,
    default="2026-01-20",
    help="Fixed date_string for tokenizer.apply_chat_template (improves reproducibility).",
)
parser.add_argument(
    "--loss_type",
    type=str,
    choices=["perplexity", "kl_divergence", "combined"],
    default="perplexity",
    help="Type of loss to use during training."
)
parser.add_argument(
    "--kl_temperature",
    type=float,
    default=1.0,
    help="Temperature for KL distillation (typical values: 1-4)."
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=24,
    help="Batch size for training."
)
parser.add_argument(
    "--grad_accumulate_steps",
    type=int,
    default=1,
    help="Batch size for training."
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=24,
    help="# of Training Steps."
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=2e-5,
    help="Learning rate for the optimizer."
)
parser.add_argument(
    "--save_every_steps",
    type=int,
    default=0,
    help="Save a checkpoint every N steps (0 = only save at the end).",
)
parser.add_argument(
    "--eval_every_steps",
    type=int,
    default=0,
    help="Run validation every N steps (0 = only run at the end).",
)
parser.add_argument(
    "--eval_max_batches",
    type=int,
    default=32,
    help="Maximum number of validation batches per evaluation pass.",
)
parser.add_argument(
    "--eval_at_start",
    action="store_true",
    default=True,
    help="Run a pre-update validation pass at step 0 (recommended).",
)
parser.add_argument("--no_eval_at_start", action="store_false", dest="eval_at_start")
parser.add_argument(
    "--warmup_step_ratio",
    type=float,
    default=0.1,
    help="Percent of total steps that should be used to warm up the linear rate scheduler."
)
parser.add_argument(
    "--lr_schedule",
    type=str,
    choices=["linear", "constant", "cosine"],
    default="linear",
    help="Learning-rate schedule (all use warmup_step_ratio for warmup).",
)
parser.add_argument(
    "--cosine_num_cycles",
    type=float,
    default=0.5,
    help="Number of cosine cycles for --lr_schedule=cosine.",
)
parser.add_argument(
    "--max_length",
    type=int,
    default=4096,
    help="Maximum sequence length for the tokenizer."
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Name of back-end device to use for PyTorch."
)
parser.add_argument(
    "--attn_implementation",
    type=str,
    choices=["auto", "eager", "sdpa", "flash_attention_2"],
    default="auto",
    help="Attention backend for HF models (auto prefers flash_attention_2 on CUDA when available).",
)
parser.add_argument(
    "--torch_dtype",
    type=str,
    choices=["auto", "bf16", "fp16", "fp32"],
    default="auto",
    help="Model weights dtype (auto uses bf16 on CUDA when supported, else fp16; CPU uses fp32).",
)
parser.add_argument(
    "--tf32",
    action="store_true",
    default=None,
    help="Enable TF32 on CUDA matmuls/convs (default: enabled for CUDA).",
)
parser.add_argument(
    "--no_tf32",
    action="store_false",
    dest="tf32",
    help="Disable TF32 on CUDA matmuls/convs.",
)
parser.add_argument(
    "--torch_compile",
    type=str,
    choices=["none", "truncated", "both"],
    default="none",
    help="Use torch.compile for the truncated model (or both teacher+student).",
)
parser.add_argument(
    "--torch_compile_mode",
    type=str,
    choices=["default", "reduce-overhead", "max-autotune"],
    default="default",
    help="torch.compile mode.",
)
parser.add_argument(
    "--scale_loss_by_grad_accumulate",
    action="store_true",
    default=True,
    help="Divide loss by grad_accumulate_steps before backward (recommended).",
)
parser.add_argument(
    "--no_scale_loss_by_grad_accumulate",
    action="store_false",
    dest="scale_loss_by_grad_accumulate",
    help="Legacy behavior: do not scale loss during gradient accumulation.",
)
parser.add_argument(
    "--wandb",
    action="store_true",
    help="Use to enable logging to wandb."
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="early-exit-tests",
    help="Weights & Biases project name (used only when --wandb is set).",
)
parser.add_argument(
    "--wandb_log_gradients",
    action="store_true",
    default=False,
    help="Log gradient histograms to W&B (slow; off by default).",
)
parser.add_argument(
    "--dataloader_num_workers",
    type=int,
    default=-1,
    help="PyTorch DataLoader workers (-1 = auto; on macOS defaults to 0).",
)
parser.add_argument(
    "--dataloader_prefetch_factor",
    type=int,
    default=2,
    help="DataLoader prefetch_factor (only used when num_workers > 0).",
)
parser.add_argument(
    "--dataloader_pin_memory",
    action="store_true",
    default=None,
    help="Enable DataLoader pin_memory (default: auto, enabled for CUDA).",
)
parser.add_argument(
    "--no_dataloader_pin_memory",
    action="store_false",
    dest="dataloader_pin_memory",
    help="Disable DataLoader pin_memory.",
)
parser.add_argument(
    "--dataloader_persistent_workers",
    action="store_true",
    default=None,
    help="Keep DataLoader workers alive (default: auto, enabled when num_workers > 0).",
)
parser.add_argument(
    "--no_dataloader_persistent_workers",
    action="store_false",
    dest="dataloader_persistent_workers",
    help="Disable DataLoader persistent_workers.",
)
parser.add_argument(
    "--lm_head_random_init",
    action="store_true",
    help="Randomly initializes the LM head to train."
)
parser.add_argument(
    "--output_dir",
    type=str,
    help="Directory for training artifacts"
)
parser.add_argument(
    "--notes",
    type=str,
    required=True,
    help="Notes about the run."
)
parser.add_argument(
    "--ft_last_transformer",
    action="store_true",
    help="finetune last transformer layer"
)
parser.add_argument(
    "--ft_head",
    action="store_true",
    help="finetune the head"
)
parser.add_argument(
    "--checkpoint_path",
    default=None,
    type=str,
    help="Path to the checkpoint to load."
)
args = parser.parse_args()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)
print(args)

# Set device to GPU if available
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if args.run_type is None:
    args.run_type = datetime.now().strftime("run_%Y%m%d_%H%M%S")
run_name = f"{args.run_type}/layer{args.target_layer}_{args.max_steps}steps_begin{int(time.time())}"
model_folder = args.model_path.split("/")[-1]
save_folder = os.path.join(args.output_dir, model_folder, run_name)
os.makedirs(save_folder, exist_ok=True)

# Map CLI-facing names to model loss_type names.
effective_loss_type = args.loss_type
if effective_loss_type == "perplexity":
    effective_loss_type = "cross_entropy"
device_type = "cuda" if str(args.device).startswith("cuda") else str(args.device)

def _resolve_torch_dtype(dtype_flag: str) -> torch.dtype:
    if dtype_flag == "bf16":
        return torch.bfloat16
    if dtype_flag == "fp16":
        return torch.float16
    if dtype_flag == "fp32":
        return torch.float32
    if dtype_flag != "auto":
        raise ValueError(f"Unknown torch_dtype: {dtype_flag}")
    if device_type == "cuda":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32

def _flash_attn_2_available() -> bool:
    try:
        from transformers.utils import is_flash_attn_2_available  # type: ignore

        return bool(is_flash_attn_2_available())
    except Exception:
        try:
            import flash_attn  # noqa: F401

            return True
        except Exception:
            return False

def _resolve_attn_implementation(attn_flag: str) -> str | None:
    if attn_flag != "auto":
        return attn_flag
    if device_type == "cuda" and _flash_attn_2_available():
        return "flash_attention_2"
    if device_type == "cuda":
        return "sdpa"
    return None

model_dtype = _resolve_torch_dtype(args.torch_dtype)
attn_impl = _resolve_attn_implementation(args.attn_implementation)

if args.tf32 is None:
    args.tf32 = device_type == "cuda"
if device_type == "cuda" and args.tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

use_autocast = device_type == "cuda" and model_dtype in (torch.bfloat16, torch.float16)
autocast_dtype = model_dtype if use_autocast else None

# DataLoader defaults.
if args.dataloader_num_workers < 0:
    if sys.platform == "darwin":
        args.dataloader_num_workers = 0
    else:
        cpu_count = os.cpu_count() or 1
        args.dataloader_num_workers = min(8, max(1, cpu_count // 2))
if args.dataloader_pin_memory is None:
    args.dataloader_pin_memory = device_type == "cuda"
if args.dataloader_persistent_workers is None:
    args.dataloader_persistent_workers = args.dataloader_num_workers > 0

# Add dataset info to run notes before any slow setup (e.g. data loading).
if args.dataset == "fineweb":
    streaming_desc = "streaming" if args.fineweb_streaming else "non-streaming"
    DATASET_DESC = f"Fineweb 10B tokens ({streaming_desc})"
elif args.dataset == "sharegpt":
    DATASET_DESC = "ShareGPT90K"
elif args.dataset == "ultrachat":
    streaming_desc = "streaming" if args.ultrachat_streaming else "non-streaming"
    DATASET_DESC = f"UltraChat 200k SFT ({streaming_desc})"
elif args.dataset == "toy":
    DATASET_DESC = "Toy strings"
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")
args.notes += f"\nDataset: {DATASET_DESC}"

# Initialize wandb before data loading.
if args.wandb:
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    wandb.init(
        project=args.wandb_project,
        name=run_name.replace("/", "__"),
        group=args.run_type,
        job_type=f"layer{args.target_layer}",
        config=vars(args),
        mode=wandb_mode,
        notes=args.notes,
    )

# Load tokenizer and create dataset.
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.pad_token = tokenizer.eos_token

# Setup for attention backend + dtype.
if attn_impl is not None:
    logger.info("Using attention implementation: %s", attn_impl)
logger.info("Using model dtype: %s", str(model_dtype).replace("torch.", ""))
model = TruncatedLlama(
    args.model_path,
    early_exit_idx=args.target_layer,
    attn_implementation=attn_impl,
    torch_dtype=model_dtype if device_type == "cuda" else None,
    use_cache=False,
    lm_head_random_init=args.lm_head_random_init,
    ft_head=args.ft_head,
    ft_last_transformer=args.ft_last_transformer,
)

if args.checkpoint_path is not None:
    checkpoint = torch.load(args.checkpoint_path)
    if args.ft_last_transformer:
        assert checkpoint["last_transformer"] is None
        model.load_from_checkpoint(checkpoint["lm_head"], None)

model.print_trainable_parameters()
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if num_trainable_params == 0:
    raise ValueError(
        "No trainable parameters. Pass `--ft_head` and/or `--ft_last_transformer` "
        "to enable training of the early-exit head and/or the last retained transformer block."
    )
model.to(args.device)
if args.torch_compile != "none":
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile requested but not available in this PyTorch build.")
    compile_kwargs = {}
    if args.torch_compile_mode != "default":
        compile_kwargs["mode"] = args.torch_compile_mode
    if args.torch_compile in ("truncated", "both"):
        logger.info("torch.compile: compiling truncated model (mode=%s)", args.torch_compile_mode)
        model.truncated_model = torch.compile(model.truncated_model, **compile_kwargs)
    if args.torch_compile == "both":
        logger.info("torch.compile: compiling reference model (mode=%s)", args.torch_compile_mode)
        model.reference_model = torch.compile(model.reference_model, **compile_kwargs)
model.train()
## Training Loop

# Uncomment to use XSUM
# train_dataset = XSUMDataset(tokenizer, max_length=args.max_length, split="train", num_proc=8)
# validation_dataset = XSUMDataset(tokenizer, max_length=args.max_length, split="validation", num_proc=8)
# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=data_collator)
# val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)

# train, test, val = get_sharegpt_dataloaders(args.batch_size, tokenizer, args.max_length, generate_labels = args.loss_type == "cross_entropy", seed=args.seed)
# DATASET_DESC = "shareGPT with non-english removed"

# Always build labels so we can report CE/perplexity during validation even for KL-only training.
generate_labels = True
if args.dataset == "fineweb":
    train, test, val = get_fineweb_dataloaders(
        args.batch_size,
        tokenizer,
        args.max_length,
        generate_labels=generate_labels,
        seed=args.seed,
        streaming=args.fineweb_streaming,
        shuffle_buffer_size=args.fineweb_shuffle_buffer_size,
        train_examples=args.fineweb_train_examples,
        val_examples=args.fineweb_val_examples,
        test_examples=args.fineweb_test_examples,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
        persistent_workers=args.dataloader_persistent_workers,
        prefetch_factor=args.dataloader_prefetch_factor,
    )
elif args.dataset == "sharegpt":
    train, test, val = get_sharegpt_dataloaders(
        args.batch_size,
        tokenizer,
        args.max_length,
        generate_labels=generate_labels,
        seed=args.seed,
        filter_non_english=args.sharegpt_filter_non_english,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
        persistent_workers=args.dataloader_persistent_workers,
        prefetch_factor=args.dataloader_prefetch_factor,
    )
elif args.dataset == "ultrachat":
    train, test, val = get_ultrachat_dataloaders(
        args.batch_size,
        tokenizer,
        args.max_length,
        generate_labels=generate_labels,
        nice_shape=True,
        seed=args.seed,
        streaming=args.ultrachat_streaming,
        shuffle_buffer_size=args.ultrachat_shuffle_buffer_size,
        train_examples=args.ultrachat_train_examples,
        val_examples=args.ultrachat_val_examples,
        test_examples=args.ultrachat_test_examples,
        date_string=args.ultrachat_date_string,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
        persistent_workers=args.dataloader_persistent_workers,
        prefetch_factor=args.dataloader_prefetch_factor,
    )
elif args.dataset == "toy":
    train, test, val = get_toy_dataloaders(
        args.batch_size,
        tokenizer,
        args.max_length,
        generate_labels=generate_labels,
        nice_shape=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
        persistent_workers=args.dataloader_persistent_workers,
        prefetch_factor=args.dataloader_prefetch_factor,
    )
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

# -----------------------
# Validation evaluation
# -----------------------
non_blocking = bool(args.dataloader_pin_memory and device_type == "cuda")

def _shift_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    shifted = torch.zeros_like(attention_mask)
    if attention_mask.size(1) > 1:
        shifted[:, :-1] = attention_mask[:, 1:]
    return shifted

def run_validation(step: int) -> dict[str, float]:
    """
    Computes token-weighted metrics on the validation loader.

    - KL is reported as nats/token (masked like training).
    - CE is reported as nats/token; ppl = exp(CE).
    - top1_acc is next-token accuracy on non-masked labels.
    """
    model.eval()

    total_kl = 0.0
    total_kl_tokens = 0
    total_ce = 0.0
    total_ce_tokens = 0
    top1_correct = 0
    top1_tokens = 0
    num_batches = 0

    compute_kl = args.loss_type in ("kl_divergence", "combined")

    with torch.inference_mode():
        for batch_idx, batch in enumerate(val):
            if batch_idx >= args.eval_max_batches:
                break
            input_ids = batch["input_ids"].to(args.device, non_blocking=non_blocking)
            attention_mask = batch["attention_mask"].to(args.device, non_blocking=non_blocking)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(args.device, non_blocking=non_blocking)

            autocast_ctx = (
                torch.autocast(device_type=device_type, dtype=autocast_dtype)
                if use_autocast
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                if compute_kl:
                    outputs = model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=None,
                        loss_type="kl_divergence",
                        kl_temperature=args.kl_temperature,
                    )
                else:
                    outputs = model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        loss_type="cross_entropy",
                        kl_temperature=args.kl_temperature,
                    )
            logits = outputs["logits"]

            if compute_kl:
                shift_mask = _shift_mask(attention_mask)
                kl_tokens = int(shift_mask.sum().item())
                if kl_tokens > 0:
                    kl_mean = float(outputs["loss"].float().item())
                    # Model scales KL by T^2 for the training objective; report unscaled KL (nats/token).
                    denom = float(args.kl_temperature) ** 2
                    if denom > 0:
                        kl_mean = kl_mean / denom
                    total_kl += kl_mean * kl_tokens
                    total_kl_tokens += kl_tokens

            if labels is not None:
                ce_tokens = int((labels != -100).sum().item())
                if ce_tokens > 0:
                    ce_sum = F.cross_entropy(
                        logits.float().view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    total_ce += float(ce_sum.item())
                    total_ce_tokens += ce_tokens

                    pred = logits.argmax(dim=-1)
                    mask = labels != -100
                    top1_correct += int(((pred == labels) & mask).sum().item())
                    top1_tokens += ce_tokens

            num_batches += 1

    model.train()

    metrics: dict[str, float] = {"val/batches": float(num_batches)}
    if total_kl_tokens > 0:
        metrics["val/kl"] = total_kl / total_kl_tokens
    if total_ce_tokens > 0:
        ce_mean = total_ce / total_ce_tokens
        metrics["val/ce"] = ce_mean
        metrics["val/ppl"] = float(math.exp(ce_mean))
    if top1_tokens > 0:
        metrics["val/top1_acc"] = top1_correct / top1_tokens
    return metrics

# Uncomment to use SlimPJ
# train_dataset = SlimPJDataset(tokenizer, max_length=max_length, split="train", num_proc=8)
# validation_dataset = SlimPJDataset(tokenizer, max_length=max_length, split="validation", num_proc=8)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
# val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

# train_dataset = load_dataset("DKYoon/SlimPajama-6B", split="train", num_proc=8)
# tokenized_dataset = train_dataset.map(lambda example: 
#                                     tokenizer(
#                                         text=example["text"],
#                                         padding=False,
#                                         max_length=args.max_length,
#                                     ),
#                                     batched=False)
# tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
# train_loader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, generate_labels=False))


# train_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train", num_proc=8)
# Weight decay, initial learning rate, set basic hyperparams.
# optimizer = model.configure_optimizers(weight_decay, args.learning_rate, args.device)
use_fused = device_type == "cuda"
try:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=use_fused,
    )
except TypeError:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
num_warmup_steps = int(args.max_steps * args.warmup_step_ratio)
if args.lr_schedule == "linear":
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_steps,
    )
elif args.lr_schedule == "constant":
    scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
    )
elif args.lr_schedule == "cosine":
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_steps,
        num_cycles=float(args.cosine_num_cycles),
    )
else:
    raise ValueError(f"Unknown lr_schedule: {args.lr_schedule}")

# Training stats.
print(f"Max Steps: {args.max_steps}")
print(f"Warmup Steps: {int(args.max_steps * args.warmup_step_ratio)}")
print(f"Effective Batch Size: {args.grad_accumulate_steps * args.batch_size}")

gradients = {}
non_pad_tok = 0
total_tok = 0
num_examples = 0
try:
    num_train_batches = len(train)
except TypeError:
    num_train_batches = None
train_iter = iter(train)

if args.eval_at_start:
    metrics = run_validation(step=0)
    metrics_str = ", ".join(f"{k}={v:.6g}" for k, v in metrics.items() if k != "val/batches")
    print(f"Pre-update validation @ step 0: {metrics_str}")
    if args.wandb:
        wandb.log(metrics, step=0)

for step in range(args.max_steps):
    step_t0 = time.time()
    # Get data tensors, move to device.
    loss_accum = 0.0
    for mini_step in range(args.grad_accumulate_steps):
        try:
            next_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train)
            next_batch = next(train_iter)
        input_ids, attention_mask, labels = next_batch["input_ids"],  next_batch["attention_mask"], next_batch["labels"]
        input_ids = input_ids.to(args.device, non_blocking=non_blocking)
        attention_mask = attention_mask.to(args.device, non_blocking=non_blocking)

        # track data stats
        non_pad_tok += int(attention_mask.sum().item())
        total_tok += int(attention_mask.numel())
        num_examples += args.batch_size

        if effective_loss_type in ("cross_entropy", "combined"):
            labels = labels.to(args.device, non_blocking=non_blocking)
        autocast_ctx = (
            torch.autocast(device_type=device_type, dtype=autocast_dtype)
            if use_autocast
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_type=effective_loss_type,
                kl_temperature=args.kl_temperature,
            )

        loss, logits = outputs["loss"], outputs["logits"]
        loss_accum += loss.detach()
        loss_to_backprop = loss / args.grad_accumulate_steps if args.scale_loss_by_grad_accumulate else loss
        loss_to_backprop.backward()
    
    should_save = (
        (args.save_every_steps > 0 and step > 0 and step % args.save_every_steps == 0)
        or step == args.max_steps - 1
    )
    should_eval = (
        (args.eval_every_steps > 0 and step > 0 and step % args.eval_every_steps == 0)
        or step == args.max_steps - 1
    )

    if should_save:
        save_path = os.path.join(save_folder, f"model_{step}_{float(loss_accum):.2f}.pt")
        d = {
            "meta": {
                "model_path": args.model_path,
                "dataset": args.dataset,
                "loss_type": args.loss_type,
                "kl_temperature": args.kl_temperature,
                "target_layer": args.target_layer,
                "ft_head": bool(args.ft_head),
                "ft_last_transformer": bool(args.ft_last_transformer),
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "grad_accumulate_steps": args.grad_accumulate_steps,
                "learning_rate": args.learning_rate,
                "warmup_step_ratio": args.warmup_step_ratio,
                "step": step,
                "time": time.time(),
            },
            "lm_head": model.truncated_model.lm_head.state_dict(),
            "last_transformer": model.early_exit_layer.state_dict() if args.ft_last_transformer else None,
        }
        torch.save(d, save_path)

    if should_eval:
        metrics = run_validation(step)
        metrics["data/ratio"] = non_pad_tok / total_tok
        metrics_str = ", ".join(f"{k}={v:.6g}" for k, v in metrics.items() if k != "val/batches")
        print(f"Validation @ step {step}: {metrics_str}")
        if args.wandb:
            wandb.log(metrics, step=step)

            if args.wandb_log_gradients:
                gradients = {}
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    grad_np = param.grad.detach().float().cpu().numpy()
                    gradients[f"gradients/{step}/{name}"] = wandb.Histogram(grad_np)
                wandb.log(gradients, step=step)

    # Clip gradients and step.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']

    # Log training stats.
    step_dt = max(1e-6, time.time() - step_t0)
    print(f"Step {step}, Train Loss: {loss_accum}, Learning Rate {current_lr}, Grad Norm: {norm}, Step Time(s): {step_dt:.3f}")
    if args.wandb:
        wandb.log({"train/loss": loss_accum, "train/grad_norm": norm, "train/lr": current_lr}, step=step)

if args.wandb:
    wandb.finish()

train_size_str = "unknown"
if num_train_batches is not None:
    train_size_str = str(num_train_batches)
print(f"non_pad: {non_pad_tok}, total_tokens: {total_tok}, #examples: {num_examples}, train_batches: {train_size_str}")

# Generate from model every 100 steps.
# .generate() uses autocast.
# if step % 100 == 0 or step == max_steps - 1:
#     prompt = "Hi! I'm a language model."
#     input_ids = tokenizer.encode(prompt, return_tensors="pt")
#     input_ids = input_ids.to(args.device)
#     model.eval()
#     with torch.no_grad():
#         for i in range(10):
#             outputs = model.generate(input_ids, max_length=20, eos_token_id=tokenizer.eos_token_id)
#             s = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#             print(s)
#             if args.wandb:
#                 wandb.log({"gen/output": s})
#     model.train()

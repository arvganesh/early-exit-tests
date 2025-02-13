import os; os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
import wandb
import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
import logging
import pickle
import math
import time
import argparse
from datetime import datetime

from truncated_llama import TruncatedLlama
from share_gpt_dataset import get_sharegpt_dataloaders

parser = argparse.ArgumentParser(description="Train a truncated Llama model with a tuned head.")
parser.add_argument(
    "--model_path",
    type=str,
    default="meta-llama/Llama-2-7b-chat-hf",
    help="Path to the model checkpoint or model identifier."
)
parser.add_argument(
    "--target_layer",
    type=int,
    default=16,
    help="The transformer layer number that you want to target."
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
    default=2.0,
    help="Temperature value for KL divergence."
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
    "--max_length",
    type=int,
    default=4096,
    help="Maximum sequence length for the tokenizer."
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="llama_tuned_head_output",
    help="Directory where the trained model and optimizer state will be saved."
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Name of back-end device to use for PyTorch."
)
parser.add_argument(
    "--wandb",
    action="store_true",
    help="Use to enable logging to wandb."
)
parser.add_argument(
    "--lm_head_random_init",
    action="store_true",
    help="Randomly initializes the LM head to train."
)
args = parser.parse_args()
print(args)

# Set device to GPU if available
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load tokenizer and create dataset.
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.pad_token = tokenizer.eos_token

# Setup for mixed precision training.
torch.set_float32_matmul_precision("high")
model = TruncatedLlama(args.model_path, 
                       early_exit_idx=args.target_layer,
                       lm_head_random_init=args.lm_head_random_init, 
                       use_flash_attn=False)
model.train()
# model = torch.compile(model) if args.device == "cuda" else model
model.to(args.device)
## Training Loop

# Uncomment to use XSUM
# train_dataset = XSUMDataset(tokenizer, max_length=args.max_length, split="train", num_proc=8)
# validation_dataset = XSUMDataset(tokenizer, max_length=args.max_length, split="validation", num_proc=8)
# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=data_collator)
# val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)

train, test, val = get_sharegpt_dataloaders(args.batch_size, tokenizer, args.max_length, generate_labels = args.loss_type == "cross_entropy")

# Uncomment to use SlimPJ
# train_dataset = SlimPJDataset(tokenizer, max_length=max_length, split="train", num_proc=8)
# validation_dataset = SlimPJDataset(tokenizer, max_length=max_length, split="validation", num_proc=8)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
# val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

# train_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train", num_proc=8)
# tokenized_dataset = train_dataset.map(lambda example: 
#                                     tokenizer(
#                                         text=example["text"],
#                                         padding=False
#                                     ),
#                                     batched=False)
# tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
# train_loader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: custom_collate_fn(batch, tokenizer))

max_lr = args.learning_rate
min_lr = max_lr * 0.1
max_steps = args.max_steps
warmup_steps = max_steps * 0.05
grad_accumulate_steps = args.grad_accumulate_steps
lr_decay = False
weight_decay = 0.01

# Linear learning rate scheduler with warmup + decay
def get_lr(it, decay=False):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if decay:
        if it > max_steps:
            return min_lr
        return min_lr + (max_lr - min_lr) * (max_steps - it) / (max_steps - warmup_steps)
    else:
        return max_lr

# Initialize wandb
if args.wandb:
    wandb.init(
        project="llama-truncated-head",
        config={
            "loss_type": args.loss_type,
            "target_layer": args.target_layer,
            "kl_temperature": args.kl_temperature,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "output_dir": args.output_dir,
            "warmup steps": warmup_steps,
            "LR decay": lr_decay,
            "weight_decay": weight_decay
        }
    )

# Weight decay, initial learning rate, set basic hyperparams.
# optimizer = model.configure_optimizers(weight_decay, args.learning_rate, args.device)
use_fused = args.device == "cuda"
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=args.learning_rate, 
                              betas=(0.9, 0.95), 
                              eps=1e-8, 
                              fused=use_fused)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=int(max_steps * 0.1),
                                            num_training_steps=max_steps)

# Training stats.
print(f"Max Steps: {max_steps}")
print(f"Warmup Steps: {warmup_steps}")
print(f"Effective Batch Size: {grad_accumulate_steps * args.batch_size}")

run_name = f"kl_div_model.{datetime.now()}"

for step in range(max_steps):
    # Get data tensors, move to device.
    loss_accum = 0.0
    for mini_step in range(grad_accumulate_steps):
        next_batch = next(iter(train))
        input_ids, attention_mask, labels = next_batch["input_ids"],  next_batch["attention_mask"], next_batch["labels"]
        input_ids, attention_mask = input_ids.to(args.device), attention_mask.to(args.device)
        labels = None
        if args.loss_type == "cross_entropy":
            labels = labels.to(args.device)
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels, loss_type=args.loss_type)

        loss, logits = outputs["loss"], outputs["logits"]
        loss = loss / grad_accumulate_steps # sum / batch_size / grad_accumulate_steps = sum / (batch_size * grad_accumulate_steps)
        loss_accum += loss.detach()
        loss.backward()

    if step % 500 == 0:
        torch.save(model.new_lm_head.state_dict(), f"../../models/{run_name}_{step}_{loss_accum:.2f}.pt")

    # Clip gradients and step.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # lr = get_lr(step, lr_decay)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']

    # Log training stats.
    print(f"Step {step}, Train Loss: {loss_accum}, Learning Rate {current_lr}")
    if args.wandb:
        wandb.log({"train/loss": loss_accum, "train/grad_norm": norm, "train/lr": current_lr})

# evaluate average perplexity over test dataset
# Get data tensors, move to device.
with torch.no_grad():
    early_exit_perplexity = 0
    expected_perplexity = 0
    total_tokens = 0
    for batch in val:
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        input_ids, attention_mask, labels = input_ids.to(args.device), attention_mask.to(args.device), labels.to(args.device)
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels, loss_type="cross_entropy", keep_og_logits=True)

        loss, logits = outputs["loss"], outputs["logits"]
        batch_tokens = attention_mask.sum().item()
        early_exit_perplexity += loss.item() * batch_tokens
        
        og_logits = outputs["og_lm_logits"]
        og_loss = F.cross_entropy(og_logits.view(-1, og_logits.size(-1)), labels.view(-1)).item()
        expected_perplexity += og_loss * batch_tokens
        total_tokens += batch_tokens


print(f"exp: {expected_perplexity / total_tokens}, actual: {early_exit_perplexity / total_tokens}")

if args.wandb:
    wandb.finish()

# Run model on validation data every 50 steps.
# if step % 500 == 0 or step == 0:
#     model.eval()
#     val_accum = 0.0
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(val_loader):
#             if batch_idx >= 10:
#                 break
#             input_ids, labels = batch["input_ids"], batch["labels"]
#             input_ids, labels = input_ids.to(args.device), labels.to(args.device)
#             with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
#                 outputs = model(input_ids, labels=labels)
#             val_loss = outputs.loss
#             val_loss /= grad_accumulate_steps
#             val_accum += val_loss.detach()
#     model.train()
#     print(f"Validation Loss: {val_accum}")
#     if args.wandb:
#         wandb.log({"val/loss": val_accum})

# save optimizer state
# if step % 1000 == 0 or step == max_steps - 1:
#     optimizer_state = optimizer.state_dict()
#     trained_params = model.model.lm_head.state_dict()
#     torch.save({
#         "optimizer_state": optimizer_state,
#         "trained_params": trained_params,
#         "step": step,
#         # "val_loss": val_accum
#     }, f"./models/xsum1/llama-trunc-{step}step")

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
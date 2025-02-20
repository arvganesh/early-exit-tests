import os; os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup
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
    "--seed",
    type=int,
    default=0,
    help="Seed for dataset split."
)
parser.add_argument(
    "--loss_type",
    type=str,
    choices=["perplexity", "kl_divergence", "combined"],
    default="perplexity",
    help="Type of loss to use during training."
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
    "--warmup_step_ratio",
    type=float,
    default=0.1,
    help="Percent of total steps that should be used to warm up the linear rate scheduler."
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
    "--wandb",
    action="store_true",
    help="Use to enable logging to wandb."
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

train, test, val = get_sharegpt_dataloaders(args.batch_size, tokenizer, args.max_length, generate_labels = args.loss_type == "cross_entropy", seed=args.seed)
DATASET_DESC = "shareGPT with non-english removed"

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

args.notes += f"\nDataset: {DATASET_DESC}"

# Initialize wandb
if args.wandb:
    wandb.init(
        project="Early Exiting Llama 3.2 1B Instruct",
        config=args,
        mode="online",
        notes=args.notes
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
                                            num_warmup_steps=int(args.max_steps * args.warmup_step_ratio),
                                            num_training_steps=args.max_steps)

# Training stats.
print(f"Max Steps: {args.max_steps}")
print(f"Warmup Steps: {int(args.max_steps * args.warmup_step_ratio)}")
print(f"Effective Batch Size: {args.grad_accumulate_steps * args.batch_size}")

run_name = f"layer{args.target_layer}_{args.max_steps}steps_begin{int(time.time())}"
model_folder = args.model_path.split("/")[1]
save_folder = os.path.join(args.output_dir, model_folder, run_name)
os.makedirs(save_folder)

gradients = {}
for step in range(args.max_steps):
    # Get data tensors, move to device.
    loss_accum = 0.0
    for mini_step in range(args.grad_accumulate_steps):
        next_batch = next(iter(train))
        input_ids, attention_mask, labels = next_batch["input_ids"],  next_batch["attention_mask"], next_batch["labels"]
        input_ids, attention_mask = input_ids.to(args.device), attention_mask.to(args.device)
        if args.loss_type == "cross_entropy":
            labels = labels.to(args.device)
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels, loss_type=args.loss_type)

        loss, logits = outputs["loss"], outputs["logits"]
        loss = loss / args.grad_accumulate_steps # sum / batch_size / grad_accumulate_steps = sum / (batch_size * grad_accumulate_steps)
        loss_accum += loss.detach()
        loss.backward()

    if step % 1000 == 0 or step == args.max_steps - 1 or step == 0:
        save_path = os.path.join(save_folder, f"model_{step}_{loss_accum:.2f}.pt")
        torch.save(model.new_lm_head.state_dict(), save_path)
        model.eval()
        val_accum = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val):
                if batch_idx >= 100:
                    break
                input_ids, attention_mask, labels = batch["input_ids"],  batch["attention_mask"], batch["labels"]
                input_ids, attention_mask = input_ids.to(args.device), attention_mask.to(args.device)
                if args.loss_type == "cross_entropy":
                    labels = labels.to(args.device)
                with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels, loss_type=args.loss_type)
                val_loss = outputs["loss"]
                val_loss /= args.grad_accumulate_steps
                val_accum += val_loss.detach()
        model.train()
        print(f"Validation Loss: {val_accum}")
        if args.wandb:
            wandb.log({"val/loss": val_accum}, step=step)

            for name, param in model.named_parameters():
                if param.grad is not None:
                    print("Name:", name)
                    gradients[f"gradients/{step}/{name}"] = wandb.Histogram(param.grad.detach().cpu().numpy())
        
            wandb.log(gradients, step=step)

    # Clip gradients and step.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']

    # Log training stats.
    print(f"Step {step}, Train Loss: {loss_accum}, Learning Rate {current_lr}, Grad Norm: {norm}")
    if args.wandb:
        wandb.log({"train/loss": loss_accum, "train/grad_norm": norm, "train/lr": current_lr}, step=step)

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import wandb
import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
import logging
from tuned_llama import LlamaWithTunedHead
from share_gpt_dataloader import ShareGPTDataset
from slim_pj_dataset import SlimPJDataset
from truncated_llama import TruncatedLlama
import pickle
import math
def train(
    model_path: str = "meta-llama/Llama-2-7b-hf",
    target_layer: int = 16,
    loss_type: Literal["perplexity", "kl_divergence", "combined"] = "perplexity",
    kl_temperature: float = 2.0,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    max_length: int = 512,
    output_dir: str = "llama_tuned_head_output",
):

    # Set device to GPU if available
    device = "cuda"
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    max_lr = learning_rate
    min_lr = max_lr * 0.1
    warmup_steps = 10
    token_total = 500000000 * 10
    num_tok_per_batch = 4227072
    max_steps = token_total // num_tok_per_batch
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)


    # Initialize wandb
    wandb.init(
        project="llama-truncated-head",
        config={
            "loss_type": loss_type,
            "target_layer": target_layer,
            "kl_temperature": kl_temperature,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "output_dir": output_dir,
            "num tokens trained on": token_total,
            "num tokens per batch": num_tok_per_batch,
            "warmup steps": warmup_steps,
        }
    )
    
    # Load tokenizer and create dataset.
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Setup for mixed precision training.
    torch.set_float32_matmul_precision("high")
    model = TruncatedLlama(model_path, num_transformer_layers=target_layer)
    model.train()
    model = torch.compile(model)
    model.to(device)

    ## Training Loop
    # Weight decay, initial learning rate, set basic hyperparams.
    optimizer = model.configure_optimizers(0.1, learning_rate, device)

    # Uncomment to use ShareGPT
    # train_dataset = ShareGPTDataset(tokenizer, max_length=max_length)

    # Uncomment to use SlimPJ
    # Average tokens per entry ~2000, but it's pretty evenly distributed between 0 - 4k.
    # Median is 2000
    train_dataset = SlimPJDataset(tokenizer, max_length=max_length, split="train", num_proc=8)
    validation_dataset = SlimPJDataset(tokenizer, max_length=max_length, split="validation", num_proc=8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    # Llama paper says the batch size was ~4M tokens.
    grad_accumulate_steps = num_tok_per_batch // (batch_size * 4096)
    val_steps = 20
    print(f"Effective Batch Size {grad_accumulate_steps * batch_size * 4096}")
    print(f"Inner Loop Steps {grad_accumulate_steps}")

    # Training stats.
    train_time_hrs = max_steps * 180 / 3600
    rate = 1.49
    print("Estimated Training Time: ", train_time_hrs, "hours")
    print(f"Cost: {train_time_hrs * rate} USD")
    for step in range(max_steps):
        # Run model on validation data every 50 steps.
        if step % 10 == 0 or step == 0:
            model.eval()
            val_accum = 0.0
            with torch.no_grad():
                for mini_step in range(val_steps):
                    next_batch = next(iter(val_loader))
                    input_ids, attention_mask, labels = next_batch["input_ids"], next_batch["attention_mask"], next_batch["labels"]
                    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss = outputs.loss
                    val_loss /= grad_accumulate_steps
                    val_accum += val_loss.detach()

            # save optimizer state
            if step % 50 == 0 or step == max_steps - 1 or step == 0:
                optimizer_state = optimizer.state_dict()
                trained_params = model.model.lm_head.state_dict()
                torch.save({
                    "optimizer_state": optimizer_state,
                    "trained_params": trained_params,
                    "step": step,
                    "val_loss": val_accum
                }, f"./models/highLR/llama-trunc-{step}step")

            wandb.log({"val/loss": val_accum})
            print(f"Step {step}, Val Loss: {val_accum}")
            model.train()
        
        # Generate from model every 100 steps.
        # .generate() uses autocast.
        if step % 10 == 0 or step == 0:
            prompt = "Hi! I'm a language model."
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids.to(device)
            model.eval()
            with torch.no_grad():
                for i in range(10):
                    outputs = model.generate(input_ids, max_length=20)
                    s = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    print(s)
                    wandb.log({"gen/output": s})
            model.train()

        # Get data tensors, move to device.
        loss_accum = 0.0
        for mini_step in range(grad_accumulate_steps):
            next_batch = next(iter(train_loader))
            input_ids, attention_mask, labels = next_batch["input_ids"], next_batch["attention_mask"], next_batch["labels"]
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss, logits = outputs["loss"], outputs["logits"]
            loss = loss / grad_accumulate_steps # sum / batch_size / grad_accumulate_steps = sum / (batch_size * grad_accumulate_steps)
            loss_accum += loss.detach()
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        optimizer.zero_grad()
        print(f"Step {step}, Norm: {norm: .4f}, Loss: {loss_accum}, lr: {lr}")
        wandb.log({"train/loss": loss_accum, "train/grad_norm": norm, "train/lr": lr})
    wandb.finish()


if __name__ == "__main__":
    # Example usage with different loss types:
    # train(loss_type="perplexity")  # Only Loss A
    # train(loss_type="kl_divergence")  # Only Loss B
    # train(loss_type="combined")  # Both losses
    train(loss_type="perplexity", target_layer=16, max_length=4096, batch_size=24, learning_rate=1e-3)
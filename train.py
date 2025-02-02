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
from peft import LoraConfig, get_peft_model
from share_gpt_dataloader import ShareGPTDataset
from truncated_llama import TruncatedLlama
import pickle
import math

# Set device to GPU if available
device = "cuda"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 5
max_steps = 100 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
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

def train(
    model_path: str = "meta-llama/Llama-2-7b-hf",
    target_layer: int = 16,
    loss_type: Literal["perplexity", "kl_divergence", "combined"] = "perplexity",
    kl_temperature: float = 2.0,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    num_epochs: int = 3,
    max_length: int = 512,
    gradient_accumulation_steps: int = 2,
    output_dir: str = "llama_tuned_head_output",
    use_lora: bool = False,
):
    # Initialize wandb
    wandb.init(
        project="llama-tuned-head",
        config={
            "loss_type": loss_type,
            "target_layer": target_layer,
            "kl_temperature": kl_temperature,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_length": max_length,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "output_dir": output_dir,
            "use_lora": use_lora,
        }
    )
    
    # Load tokenizer and create dataset
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = ShareGPTDataset(tokenizer, max_length=max_length)
    model = TruncatedLlama(model_path, num_transformer_layers=target_layer)
    model.to(device)
    model.train()
    torch.set_float32_matmul_precision("high")
    
    # Train
    """
    effective_batch_size = ~4,000,000 => 4194304
    actual_batch = 4096
    dataset size = 1900 * 1400 = ~126,000,000
    each step = 4s, -> 4000s per batch (about 1 hour)
    25 hours per epoch ish?
    """

    optimizer = model.configure_optimizers(0.1, learning_rate, device)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    grad_accumulate_steps = 4194304 // 4096
    # grad_accumulate_steps = 2
    print(f"Effective Batch Size {grad_accumulate_steps * batch_size * 4096}")
    loss_accum = 0.0
    for step in range(max_steps):
        if step % 8 == 0 or step == max_steps - 1 or step == 0:
            trained_params = model.model.lm_head.state_dict()
            torch.save(trained_params, f"./truncated_llama/llama-trunc-{step}step")

        # Get data tensors, move to device.
        for mini_step in range(grad_accumulate_steps):
            next_batch = next(iter(train_loader))
            input_ids, attention_mask, labels = next_batch["input_ids"], next_batch["attention_mask"], next_batch["labels"]
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss, logits = outputs["loss"], outputs["logits"]
            loss = loss / grad_accumulate_steps
            loss_accum += loss.detach()
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        optimizer.zero_grad()
        print(f"Step {step}, Norm: {norm: .4f}, Loss: {loss_accum}")

    wandb.finish()


if __name__ == "__main__":
    # Example usage with different loss types:
    # train(loss_type="perplexity")  # Only Loss A
    # train(loss_type="kl_divergence")  # Only Loss B
    # train(loss_type="combined")  # Both losses
    train(loss_type="perplexity", target_layer=16, use_lora=False, max_length=4096, batch_size=1)
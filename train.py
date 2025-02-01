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
# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShareGPTDataset(Dataset):
    def __init__(self, tokenizer: LlamaTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = self.load_and_preprocess_dataset()

    def load_and_preprocess_dataset(self):
        preprocessed_dataset = []
        dataset = load_dataset("liyucheng/ShareGPT90K")["train"]
        for item in dataset:
            conversation = ""
            for turn, content in zip(item["conversations"]["from"], item["conversations"]["value"]):
                conversation += f"{turn}: {content}\n"
                if len(conversation) > self.max_length:
                    break
            
            encodings = self.tokenizer(
                conversation.strip(),
                max_length=self.max_length,
                padding="max_length", 
                truncation=True,
                return_tensors="pt" # Pytorch tensors
            )
            
            input_ids = encodings["input_ids"].squeeze()
            attention_mask = encodings["attention_mask"].squeeze()
        
            preprocessed_dataset.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone()
            })
        return preprocessed_dataset

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.dataset[idx]

class CustomTrainer(Trainer):
    def log(self, logs: Dict[str, float], start_time=None):
        """Custom logging to track individual loss components"""
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 4)

        if "metrics" in logs:
            metrics = logs.pop("metrics")
            logs.update(metrics)

        super().log(logs)

def train(
    model_path: str = "meta-llama/Llama-2-7b-hf",
    target_layer: int = 16,
    loss_type: Literal["perplexity", "kl_divergence", "combined"] = "perplexity",
    kl_temperature: float = 2.0,
    batch_size: int = 12,
    learning_rate: float = 1e-4,
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

    print("Processed Dataset")
    
    # # Create model with tuned prediction head
    model = LlamaWithTunedHead(
        model_path,
        target_layer,
        loss_type=loss_type,
        kl_temperature=kl_temperature,
        dtype=torch.bfloat16,
    )

    if use_lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["target_lm_head"],
            use_rslora=True,
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        report_to="wandb",
        run_name=f"llama-tuned-head-{loss_type}-{target_layer}-{kl_temperature}-{batch_size}-{learning_rate}-{num_epochs}-{max_length}-{gradient_accumulation_steps}"
    )
    
    # Initialize trainer with custom logging
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda data: {
            "input_ids": torch.stack([x["input_ids"] for x in data]),
            "attention_mask": torch.stack([x["attention_mask"] for x in data]),
            "labels": torch.stack([x["labels"] for x in data]),
        },
    )
    
    # # Train
    trainer.train()
    
    # # Save final model
    trainer.save_model()
    wandb.finish()

if __name__ == "__main__":
    # Example usage with different loss types:
    # train(loss_type="perplexity")  # Only Loss A
    # train(loss_type="kl_divergence")  # Only Loss B
    # train(loss_type="combined")  # Both losses
    train(loss_type="perplexity", use_lora=True)
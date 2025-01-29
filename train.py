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

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaWithTunedHead(nn.Module):
    def __init__(
        self, 
        base_model_path: str, 
        target_layer: int,
        loss_type: Literal["perplexity", "kl_divergence", "combined"] = "perplexity",
        kl_temperature: float = 2.0,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            torch_dtype=dtype
        )
        self.target_layer = target_layer
        self.loss_type = loss_type
        self.kl_temperature = kl_temperature
        
        # Create a copy of the LM head for the target layer
        self.target_lm_head = nn.Linear(
            self.model.lm_head.in_features,
            self.model.lm_head.out_features,
            bias=False,
            dtype=dtype
        )

        # Initialize with the same weights as the original LM head
        self.target_lm_head.weight.data = self.model.lm_head.weight.data.clone()
        
        # Freeze all parameters except the target LM head and target layer
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        
        # Ensure the target LM head is trainable
        for param in self.target_lm_head.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        # Get intermediate hidden states and final logits
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        
        loss = None
        metrics = {}
        
        if labels is not None:
            # Get logits from target layer
            target_hidden = outputs.hidden_states[self.target_layer]
            target_logits = self.target_lm_head(target_hidden)
            
            # Prepare shifted sequences for loss computation
            shift_logits = target_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            full_logits = outputs.logits[..., :-1, :].contiguous()
            
            # Compute losses based on selected type
            if self.loss_type in ["perplexity", "combined"]:
                loss_fct = nn.CrossEntropyLoss()
                loss_a = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                shift_labels.view(-1))
                metrics["perplexity_loss"] = loss_a.item()
                
                if self.loss_type == "perplexity":
                    loss = loss_a
            
            if self.loss_type in ["kl_divergence", "combined"]:
                kl_loss = nn.KLDivLoss(reduction="batchmean")
                loss_b = kl_loss(
                    F.log_softmax(shift_logits / self.kl_temperature, dim=-1),
                    F.softmax(full_logits / self.kl_temperature, dim=-1)
                )
                metrics["kl_loss"] = loss_b.item()
                
                if self.loss_type == "kl_divergence":
                    loss = loss_b
            
            if self.loss_type == "combined":
                loss = loss_a + loss_b
        
        return {
            "loss": loss,
            "target_logits": target_logits if labels is not None else None,
            "full_logits": outputs.logits,
            "metrics": metrics
        }

class ShareGPTDataset(Dataset):
    def __init__(self, tokenizer: LlamaTokenizer, max_length: int = 512):
        self.dataset = load_dataset("liyucheng/ShareGPT90K")["train"]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        
        # Format conversation with roles
        conversation = ""
        for turn, content in zip(item["conversations"]["from"], item["conversations"]["value"]):
            conversation += f"{turn}: {content}\n"
            
        encodings = self.tokenizer(
            conversation.strip(),
            max_length=self.max_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt" # Pytorch tensors
        )
        
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

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
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    num_epochs: int = 3,
    max_length: int = 512,
    gradient_accumulation_steps: int = 2,
    output_dir: str = "llama_tuned_head_output",
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
        }
    )
    
    # Load tokenizer and create dataset
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = ShareGPTDataset(tokenizer, max_length=max_length)
    
    # # Create model with tuned prediction head
    model = LlamaWithTunedHead(
        model_path,
        target_layer,
        loss_type=loss_type,
        kl_temperature=kl_temperature,
        dtype=torch.bfloat16,
    )

    # lora_config = LoraConfig(
    #     r=8,
    #     lora_alpha=32,
    #     target_modules=["target_lm_head"],
    #     use_rslora=True,
    #     lora_dropout=0.1,
    #     bias="none",
    # )
    
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
    
    # # Initialize trainer with custom logging
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
    train(loss_type="perplexity")
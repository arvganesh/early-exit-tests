from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torch
from typing import Dict
class ShareGPTDataset(Dataset):
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset("liyucheng/ShareGPT90K")["train"]

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        conversation = ""
        item = self.dataset[idx]
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

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }
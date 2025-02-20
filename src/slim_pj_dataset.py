from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torch
from typing import Dict
class SlimPJDataset(Dataset):
    def __init__(self, tokenizer, max_length: int = 512, split: str = "train", num_proc: int = 8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset("DKYoon/SlimPajama-6B", split=split, num_proc=num_proc)

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_text = self.dataset[idx]["text"]
        if len(input_text) > self.max_length:
            input_text = input_text[:self.max_length]
        
        encodings = self.tokenizer(
            input_text.strip(),
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt" # Pytorch tensors
        )
        
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()

        return {
            "input_ids": input_ids[:-1],
            "attention_mask": attention_mask[:-1],
            "labels": input_ids.clone()[1:]
        }

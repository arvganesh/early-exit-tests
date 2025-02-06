from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torch
from typing import Dict
class XSUMDataset(Dataset):
    def __init__(self, tokenizer, max_length: int = 512, split: str = "train", num_proc: int = 8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset("EdinburghNLP/xsum", split=split, num_proc=num_proc)

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_text = self.dataset[idx]["text"]
        if len(input_text) > self.max_length:
            input_text = input_text[:self.max_length]
        
        encodings = self.tokenizer(
            input_text.strip(),
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
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset
import torch
from typing import Dict
class XSUMDataset(IterableDataset):
    def __init__(self, tokenizer, max_length: int = 512, split: str = "train", num_proc: int = 8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset("EdinburghNLP/xsum", split=split, num_proc=num_proc)

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __iter__(self):
        for i in range(len(self.dataset)):
            doc_text = self.dataset[i]["document"].strip()
            sum_text = (" TL;DR: " + self.dataset[i]["summary"]).strip()
            if len(doc_text + sum_text) <= self.max_length:
                doc_encodings = self.tokenizer(
                    doc_text,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt" # Pytorch tensors
                )

                summary_encodings = self.tokenizer(
                    " TL;DR: " + self.dataset[i]["summary"].strip(),
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt" # Pytorch tensors
                )

                input_ids = doc_encodings["input_ids"].squeeze() + summary_encodings["input_ids"].squeeze()
                attention_mask = doc_encodings["attention_mask"].squeeze()
                labels = input_ids.clone()[1:]
                labels = torch.cat((labels, torch.tensor([-100])))

                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        doc_text = self.dataset[idx]["document"]
        summary_text = self.dataset[idx]["summary"]
        if len(doc_text) > self.max_length:
            doc_text = doc_text[:self.max_length]
        
        # input_encodings = self.tokenizer(
        #     input_text.strip(),
        #     max_length=self.max_length,
        #     padding="max_length", 
        #     truncation=True,
        #     return_tensors="pt" # Pytorch tensors
        # )

        # summary_text = self.dataset[idx]["summary"]
        # summary_encodings = self.tokenizer(
        #     summary_text.strip(),
        #     max_length=self.max_length,
        #     padding="max_length", 
        #     truncation=True,
        #     return_tensors="pt" # Pytorch tensors
        # )
        
        # input_ids = input_encodings["input_ids"].squeeze()
        # summary_ids = summary_encodings["input_ids"].squeeze()  
        # attention_mask = input_encodings["attention_mask"].squeeze()

        # return {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "labels": summary_ids
        # }
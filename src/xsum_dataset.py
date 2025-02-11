from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset
import torch
from typing import Dict
class XSUMDataset(Dataset):
    def __init__(self, tokenizer, max_length: int = 512, split: str = "train", num_proc: int = 8, should_pad=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = self.preprocess(load_dataset("EdinburghNLP/xsum", split=split, num_proc=num_proc))
        self.ignore_in_loss_token_id = -100
        self.should_pad = should_pad

    def __len__(self) -> int:
        return len(self.dataset)

    def preprocess(self, dataset):
        # remove indices where doc + summary > self.max_length
        dataset = dataset.filter(lambda x: 
            500 < len(x["document"].strip()) + len(x["summary"].strip()) <= self.max_length and len(x["document"].strip()) > 10 and len(x["summary"].strip()) > 10
        )
        return dataset
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        doc_text = (self.dataset[idx]["document"] + " TL;DR: ").strip()
        sum_text = self.dataset[idx]["summary"].strip()
        doc_encodings = self.tokenizer(
            doc_text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True, # Adds BOS Token.
            return_tensors="pt" # Pytorch tensors
        )

        summary_encodings = self.tokenizer(
            sum_text,
            max_length=self.max_length - 1,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt" # Pytorch tensors
        )

        """
        (Prompt, Target) pair looks like:

        # Labels should be:
        structure: <bos> <document> " TL;DR: " <summary> <eos> <pad>
        input_ids: <bos> <d1> ... <dN> <tl1> <tl2> <s1> ... <sM> <eos> <pad>
        labels:    -100 -100 ... -100  -100  <s1>  <s2> ... <eos> -100 -100
        """

        summary_encodings["input_ids"] = torch.cat((summary_encodings["input_ids"].squeeze(), torch.tensor([self.tokenizer.eos_token_id])))
        doc_tokens = doc_encodings["input_ids"].squeeze() 
        summary_tokens = summary_encodings["input_ids"].squeeze()
        num_pad = self.max_length - len(summary_tokens) - len(doc_tokens)

        input_ids = torch.cat((doc_tokens, summary_tokens, torch.tensor([self.tokenizer.pad_token_id] * num_pad)))
        labels = torch.tensor([self.ignore_in_loss_token_id] * self.max_length)
        labels[len(doc_tokens) - 1:len(doc_tokens) + len(summary_tokens) - 1] = summary_tokens.clone()

        assert input_ids != None
        assert labels != None

        # Make sure labels are shifted.
        return     {
            "input_ids": input_ids,
            "labels": labels,
            # "attention_mask": attention_mask
        }
    
    # def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    #     doc_text = self.dataset[idx]["document"]
    #     summary_text = self.dataset[idx]["summary"]
    #     if len(doc_text) > self.max_length:
    #         doc_text = doc_text[:self.max_length]
        
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

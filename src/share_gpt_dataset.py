"""
ShareGPT specific helper functions.
"""

from data_utils import custom_collate_fn
from datasets import load_dataset
from torch.utils.data import DataLoader

def tokenize_sharegpt_examples(example, tokenizer, max_length):
    conversation = ""
    for turn, content in zip(example["conversations"]["from"], example["conversations"]["value"]):
        conversation += f"{turn}: {content}\n"
        if len(conversation) > max_length:
            conversation = conversation[:max_length]
            break
    return tokenizer(text=conversation, padding=False)

def create_sharegpt_train_test_val(train_size=0.8, test_size=0.1):
    assert train_size != 0
    assert test_size != 0

    seed = 42
    dataset = load_dataset("liyucheng/ShareGPT90K", split="train", num_proc=4)
    split_datasets = dataset.train_test_split(train_size=train_size, seed=seed)
    train, non_train = split_datasets["train"], split_datasets["test"]

    train_val_datasets = non_train.train_test_split(train_size=(test_size / (1 - train_size)), seed=seed)
    test, val = train_val_datasets["train"], train_val_datasets["test"]
    return train, test, val

def get_sharegpt_dataloaders(batch_size, tokenizer):
    train, test, val = create_sharegpt_train_test_val()
    train = train.map(tokenize_sharegpt_examples, batched=False).set_format(type="torch", columns=["input_ids", "attention_mask"])
    test = test.map(tokenize_sharegpt_examples, batched=False).set_format(type="torch", columns=["input_ids", "attention_mask"])
    val = val.map(tokenize_sharegpt_examples, batched=False).set_format(type="torch", columns=["input_ids", "attention_mask"])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: custom_collate_fn(batch, tokenizer))
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: custom_collate_fn(batch, tokenizer))
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: custom_collate_fn(batch, tokenizer))
    return train_loader, test_loader, val_loader
"""
FineWeb specific helper functions.
"""

from data_utils import custom_collate_fn
from datasets import load_dataset
from torch.utils.data import DataLoader
from langdetect import detect, LangDetectException

def tokenize_fineweb_examples(example, tokenizer, max_length):
    # Adapt this function based on the structure of the fineweb dataset
    # Assuming the dataset has a 'text' field that contains the content
    text = example.get("text", "")
    if len(text) > max_length:
        text = text[:max_length]
    
    outputs = tokenizer(text=text, padding=False)
    return outputs

def create_fineweb_train_test_val(train_size=0.8, test_size=0.1, seed=0):
    assert train_size != 0
    assert test_size != 0

    dataset = load_dataset("HuggingFaceFW/fineweb", split="train", num_proc=4)
    split_datasets = dataset.train_test_split(train_size=train_size, seed=seed)
    train, non_train = split_datasets["train"], split_datasets["test"]

    train_val_datasets = non_train.train_test_split(train_size=(test_size / (1 - train_size)), seed=seed)
    test, val = train_val_datasets["train"], train_val_datasets["test"]
    return train, test, val

def get_fineweb_dataloaders(batch_size, tokenizer, max_length, train_size=0.8, test_size=0.1, val_size=0.1, seed=0, generate_labels=True, nice_shape=True):
    assert train_size + test_size + val_size == 1.0
    train, test, val = create_fineweb_train_test_val(train_size=train_size, test_size=test_size, seed=seed)

    def tokenizer_wrapper(example):
        return tokenize_fineweb_examples(example, tokenizer, max_length)

    # Remove non english items.
    # Adjust this depending on the structure of the fineweb dataset
    train = train.filter(lambda x: is_english(x.get("text", "")))
    val = val.filter(lambda x: is_english(x.get("text", "")))
    test = test.filter(lambda x: is_english(x.get("text", "")))
    
    train = train.map(tokenizer_wrapper, batched=False)
    test = test.map(tokenizer_wrapper, batched=False)
    val = val.map(tokenizer_wrapper, batched=False)

    train.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val.set_format(type="torch", columns=["input_ids", "attention_mask"])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, generate_labels=generate_labels, nice_shape=nice_shape))
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, generate_labels=True, nice_shape=nice_shape))
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, generate_labels=True, nice_shape=nice_shape))
    return train_loader, test_loader, val_loader

def is_english(text):
    """Return True if the detected language of the text is English."""
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False
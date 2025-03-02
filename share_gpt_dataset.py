"""
ShareGPT specific helper functions.
"""

from data_utils import custom_collate_fn
from datasets import load_dataset
from torch.utils.data import DataLoader
from langdetect import detect, LangDetectException

#def tokenize_sharegpt_examples(example, tokenizer, max_length):
#    conversation = ""
#    for turn, content in zip(example["conversations"]["from"], example["conversations"]["value"]):
#        conversation += f"{turn}: {content}\n"
#        if len(conversation) > max_length:
#            conversation = conversation[:max_length]
#            break
#    outputs = tokenizer(text=conversation, padding=False)
#    return outputs

def tokenize_sharegpt_examples(example, tokenizer, max_length):
    input_ids = []
    loss_mask = []
    attn_mask = []

    for turn, content in zip(example["conversations"]["from"], example["conversations"]["value"]):
        text = f"{turn}: {content}\n"
        encoded = tokenizer(text, add_special_tokens=False)
        tokens = encoded["input_ids"]
        
        # Set attention mask: 0 for human, 1 for GPT/assistant tokens
        if turn.lower() == "human":
            turn_mask = [0] * len(tokens)
        else:
            turn_mask = [1] * len(tokens)
        
        # Truncate if adding this turn would exceed max_length
        if len(input_ids) + len(tokens) > max_length:
            remaining = max_length - len(input_ids)
            input_ids.extend(tokens[:remaining])
            loss_mask.extend(turn_mask[:remaining])
            attn_mask.extend([1] * remaining)
            break
        else:
            input_ids.extend(tokens)
            loss_mask.extend(turn_mask)
            attn_mask.extend([1] * len(turn_mask))
    
    example = {"input_ids": input_ids, "loss_mask": loss_mask, "attention_mask": attn_mask}
    return example

def create_sharegpt_train_test_val(train_size=0.8, test_size=0.1, seed=0):
    assert train_size != 0
    assert test_size != 0

    dataset = load_dataset("liyucheng/ShareGPT90K", split="train", num_proc=4)
    split_datasets = dataset.train_test_split(train_size=train_size, seed=seed)
    train, non_train = split_datasets["train"], split_datasets["test"]

    train_val_datasets = non_train.train_test_split(train_size=(test_size / (1 - train_size)), seed=seed)
    test, val = train_val_datasets["train"], train_val_datasets["test"]
    return train, test, val

def get_sharegpt_dataloaders(batch_size, tokenizer, max_length, train_size=0.8, test_size=0.1, val_size=0.1, seed=0, generate_labels=True, nice_shape=True):
    assert train_size + test_size + val_size == 1.0
    train, test, val = create_sharegpt_train_test_val(train_size=train_size, test_size=test_size, seed=seed)

    def tokenizer_wrapper(example):
        return tokenize_sharegpt_examples(example, tokenizer, max_length)

    # Remove non english items.
    train = train.filter(lambda x: is_english(x["conversations"]["value"][0]))
    val = val.filter(lambda x: is_english(x["conversations"]["value"][0]))
    test = test.filter(lambda x: is_english(x["conversations"]["value"][0]))
    train = train.map(tokenizer_wrapper, batched=False)
    test = test.map(tokenizer_wrapper, batched=False)
    val = val.map(tokenizer_wrapper, batched=False)

    print(test)

    cols = ["input_ids", "attention_mask", "loss_mask"]
    train.set_format(type="torch", columns=cols)
    test.set_format(type="torch", columns=cols)
    val.set_format(type="torch", columns=cols)

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

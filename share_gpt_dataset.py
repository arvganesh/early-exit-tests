"""
ShareGPT specific helper functions.
"""

from data_utils import custom_collate_fn
from datasets import load_dataset
from torch.utils.data import DataLoader
try:
    # Optional dependency: only needed when `filter_non_english=True`.
    from langdetect import detect, LangDetectException  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    detect = None

    class LangDetectException(Exception):
        pass

def tokenize_sharegpt_examples(example, tokenizer, max_length):
    conversation = ""
    for turn, content in zip(example["conversations"]["from"], example["conversations"]["value"]):
        conversation += f"{turn}: {content}\n"
    outputs = tokenizer(
        text=conversation,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )
    return outputs

def create_sharegpt_train_test_val(train_size=0.8, test_size=0.1, seed=0):
    assert train_size != 0
    assert test_size != 0

    dataset = load_dataset("liyucheng/ShareGPT90K", split="train", num_proc=4)
    split_datasets = dataset.train_test_split(train_size=train_size, seed=seed)
    train, non_train = split_datasets["train"], split_datasets["test"]

    train_val_datasets = non_train.train_test_split(train_size=(test_size / (1 - train_size)), seed=seed)
    test, val = train_val_datasets["train"], train_val_datasets["test"]
    return train, test, val

def get_sharegpt_dataloaders(
    batch_size,
    tokenizer,
    max_length,
    train_size=0.8,
    test_size=0.1,
    val_size=0.1,
    seed=0,
    generate_labels=True,
    nice_shape=True,
    *,
    filter_non_english: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
):
    assert train_size + test_size + val_size == 1.0
    train, test, val = create_sharegpt_train_test_val(train_size=train_size, test_size=test_size, seed=seed)

    def tokenizer_wrapper(example):
        return tokenize_sharegpt_examples(example, tokenizer, max_length)

    # Optional (slow) language filtering.
    if filter_non_english:
        if detect is None:
            raise ModuleNotFoundError(
                "langdetect is required for ShareGPT language filtering. "
                "Install it or pass filter_non_english=False / --no_sharegpt_filter_non_english."
            )
        train = train.filter(lambda x: is_english(x["conversations"]["value"][0]))
        val = val.filter(lambda x: is_english(x["conversations"]["value"][0]))
        test = test.filter(lambda x: is_english(x["conversations"]["value"][0]))
    train = train.map(tokenizer_wrapper, batched=False)
    test = test.map(tokenizer_wrapper, batched=False)
    val = val.map(tokenizer_wrapper, batched=False)

    train.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val.set_format(type="torch", columns=["input_ids", "attention_mask"])

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train,
        collate_fn=lambda batch: custom_collate_fn(
            batch, tokenizer, generate_labels=generate_labels, nice_shape=nice_shape
        ),
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, generate_labels=True, nice_shape=nice_shape),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, generate_labels=True, nice_shape=nice_shape),
        **loader_kwargs,
    )
    return train_loader, test_loader, val_loader

def is_english(text):
    """Return True if the detected language of the text is English."""
    if detect is None:
        # Should only happen if called while langdetect is missing; callers should gate on it.
        return True
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

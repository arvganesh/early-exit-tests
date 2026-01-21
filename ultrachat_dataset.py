"""
UltraChat 200k specific helper functions.

Dataset: HuggingFaceH4/ultrachat_200k (splits: train_sft, test_sft, train_gen, test_gen)
We use the SFT splits for chat-style distillation and evaluation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader


def get_ultrachat_dataloaders(
    batch_size: int,
    tokenizer,
    max_length: int,
    *,
    generate_labels: bool = True,
    nice_shape: bool = True,
    seed: int = 0,
    streaming: bool = True,
    shuffle_buffer_size: int = 10_000,
    train_examples: int | None = None,
    val_examples: int = 2048,
    test_examples: int = 2048,
    date_string: str = "2026-01-20",
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
):
    """
    Returns (train_loader, test_loader, val_loader) for UltraChat SFT splits.

    Notes:
    - Uses streaming by default so we only tokenize what we iterate over.
    - Uses tokenizer.apply_chat_template (deterministic via date_string).
    """

    train_split = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        streaming=streaming,
    )
    test_split = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="test_sft",
        streaming=streaming,
    )

    # Shuffle training stream/dataset.
    shuffle_kwargs: dict[str, Any] = {"seed": seed}
    if streaming:
        shuffle_kwargs["buffer_size"] = shuffle_buffer_size
    train_split = train_split.shuffle(**shuffle_kwargs)

    def tokenize_example(example: dict[str, Any]) -> dict[str, Any]:
        messages = example.get("messages")
        if messages is None:
            raise KeyError("UltraChat example missing `messages` field.")
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=max_length,
            date_string=date_string,
        )
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

    tokenized_train = train_split.map(
        tokenize_example,
        remove_columns=getattr(train_split, "column_names", None),
    )
    tokenized_test = test_split.map(
        tokenize_example,
        remove_columns=getattr(test_split, "column_names", None),
    )

    # Keep val/test slices small and cheap (especially for streaming).
    val_dataset = tokenized_test.take(val_examples)
    test_dataset = tokenized_test.skip(val_examples).take(test_examples)
    train_dataset = tokenized_train
    if train_examples is not None:
        train_dataset = train_dataset.take(train_examples)

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        padded = tokenizer.pad(batch, padding=True, return_tensors="pt")
        input_ids = padded["input_ids"]
        attention_mask = padded["attention_mask"]

        if nice_shape:
            seq_len = int(input_ids.size(1))
            target_len = 1 << (seq_len - 1).bit_length()
            if target_len != seq_len:
                pad = target_len - seq_len
                input_ids = F.pad(input_ids, (0, pad), value=tokenizer.pad_token_id)
                attention_mask = F.pad(attention_mask, (0, pad), value=0)

        labels = None
        if generate_labels:
            pad_tokens = torch.full((input_ids.size(0), 1), tokenizer.pad_token_id, dtype=input_ids.dtype)
            labels = torch.cat((input_ids[:, 1:], pad_tokens), dim=1)
            shift_mask = torch.zeros_like(attention_mask)
            if attention_mask.size(1) > 1:
                shift_mask[:, :-1] = attention_mask[:, 1:]
            labels = labels.masked_fill(shift_mask == 0, -100)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    loader_kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)
    return train_loader, test_loader, val_loader

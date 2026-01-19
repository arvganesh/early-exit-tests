import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

def get_fineweb_dataloaders(
    batch_size,
    tokenizer,
    max_length,
    generate_labels: bool = False,
    seed: int = 42,
    *,
    streaming: bool = True,
    shuffle_buffer_size: int = 10_000,
    train_examples: int | None = None,
    val_examples: int = 512,
    test_examples: int = 512,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
):
    """
    Load and prepare the 10B Fineweb dataset for training.
    Returns train, test, val dataloaders.
    """
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=streaming,
    )

    # For streaming datasets, shuffle uses a buffer; for non-streaming, it's a full shuffle.
    shuffle_kwargs = {"seed": seed}
    if streaming:
        shuffle_kwargs["buffer_size"] = shuffle_buffer_size
    dataset = dataset.shuffle(**shuffle_kwargs)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )

    # For streaming, .map is lazy; for non-streaming, it's eager and can take a long time.
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=256,
        remove_columns=["text", "url", "date"],
    )

    # Keep val/test small and cheap: avoid skipping millions of rows.
    val_dataset = tokenized_dataset.take(val_examples)
    test_dataset = tokenized_dataset.skip(val_examples).take(test_examples)
    train_dataset = tokenized_dataset.skip(val_examples + test_examples)
    if train_examples is not None:
        train_dataset = train_dataset.take(train_examples)
    
    # Create custom collate function
    def collate_fn(batch):
        padded = tokenizer.pad(batch, padding=True, return_tensors="pt")
        input_ids = padded["input_ids"]
        attention_mask = padded["attention_mask"]
        
        labels = None
        if generate_labels:
            pad_tokens = torch.full((input_ids.size(0), 1), tokenizer.pad_token_id, dtype=input_ids.dtype)
            labels = torch.cat((input_ids[:, 1:], pad_tokens), dim=1)
            shift_mask = torch.zeros_like(attention_mask)
            if attention_mask.size(1) > 1:
                shift_mask[:, :-1] = attention_mask[:, 1:]
            labels = labels.masked_fill(shift_mask == 0, -100)
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,  # already shuffled above (or buffered shuffle if streaming)
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        **loader_kwargs,
    )
    
    val_loader = DataLoader(
        val_dataset,
        **loader_kwargs,
    )
    
    test_loader = DataLoader(
        test_dataset,
        **loader_kwargs,
    )
    
    return train_loader, test_loader, val_loader

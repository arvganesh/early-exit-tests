import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

def get_fineweb_dataloaders(batch_size, tokenizer, max_length, generate_labels=False, seed=42):
    """
    Load and prepare the 10B Fineweb dataset for training.
    Returns train, test, val dataloaders.
    """
    # Load the Fineweb dataset (10B tokens version)
    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=False)
    
    # Shuffle and take a subset for validation and testing
    dataset = dataset.shuffle(seed=seed)
    
    # Define tokenization function
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        
        if generate_labels:
            result["labels"] = result["input_ids"].copy()
        else:
            # For perplexity or kl_divergence loss, labels are None
            result["labels"] = None
            
        return result
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=["text", "url", "date"]
    )
    
    # Create splits - for a dataset this large, we'll use streaming
    # This creates iterators that can be used indefinitely
    train_dataset = tokenized_dataset.take(9_800_000)  # Most for training
    val_dataset = tokenized_dataset.skip(9_800_000).take(100_000)  # 100k for validation
    test_dataset = tokenized_dataset.skip(9_900_000).take(100_000)  # 100k for testing
    
    # Create custom collate function
    def collate_fn(batch):
        # Process batch and handle variable lengths
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        
        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in input_ids], 
            batch_first=True, 
            padding_value=tokenizer.pad_token_id
        )
        
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in attention_mask], 
            batch_first=True, 
            padding_value=0
        )
        
        if generate_labels:
            labels = [item["labels"] for item in batch]
            labels = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x) for x in labels], 
                batch_first=True, 
                padding_value=-100  # Standard ignore index for CrossEntropyLoss
            )
        else:
            labels = None
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn,
        shuffle=False  # Already shuffled the dataset
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn,
        shuffle=False
    )
    
    return train_loader, test_loader, val_loader 
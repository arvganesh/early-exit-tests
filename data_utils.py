import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

def custom_collate_fn(batch, tokenizer, generate_labels=True):
    """
    Expect the batch to look like this:
    [
    {"input_ids": [torch.Tensor], "attention_mask": torch.Tensor},
    ...
    {},
    ]

    Return: A dictionary of 3 tensors that have len(batch) elements, corresponding to the three keys.
    Additionally, this function will dynamically pad elements in the batch so they are the same shape.

    'input_ids': [[128000, 284, 86262, 88, 4298, 66416, 14767, 284, 720]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1]]
    """

    # Pad to the max length item for the entire batch, rounded up to nearest power of two.
    batch_length = max([len(elem["input_ids"]) for elem in batch])
    batch_length = 2 ** math.ceil(math.log(batch_length, 2))

    input_ids = torch.stack([F.pad(elem["input_ids"], 
                            (0, batch_length - len(elem["input_ids"])),
                            mode="constant",
                            value=tokenizer.pad_token_id) for elem in batch], dim=0)
    attention_mask = torch.stack([F.pad(elem["attention_mask"],
                                 (0, batch_length - len(elem["attention_mask"])),
                                 mode="constant",
                                 value=0) for elem in batch], dim=0)

    attention_mask = attention_mask[:, 1:]
    pad_mask = torch.zeros(len(batch), 1)
    attention_mask = torch.cat((attention_mask, pad_mask), dim=1)

    pad_tokens = torch.full((len(batch), 1), tokenizer.pad_token_id)

    labels = None
    if generate_labels:
        labels = input_ids.clone()[:, 1:] # Remove first token from labels.
        labels = torch.cat((labels, pad_tokens), dim=1) # Add another padding token
        labels = labels.masked_fill(attention_mask == 0, -100)
        assert labels.shape == input_ids.shape
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def get_toy_dataloaders(batch_size, tokenizer, max_length, generate_labels = True):
    dataset = [
        "abcdefghij",
        "klmnopqrst",
        "uvwxyzabcd",
        "HelloWorld!",
        "DataScience",
        "MachineLearn",
        "DeepLearning",
        "ArtificialNN",
        "ChatGPTModel",
        "OpenAI_Rocks",
        "QuantumLeap",
        "NeuralNet123",
        "CodingIsFun!",
        "StackOverflow",
        "PythonRocks!",
        "LambdaCalculus",
        "TensorCompute",
        "VariableNames",
        "AlgorithmX",
        "SynthesizeData"
    ]


    def tokenizer_wrapper(example):
        inputs = tokenizer(example, padding=False)
        inputs["input_ids"] = torch.tensor(inputs["input_ids"])
        inputs["attention_mask"] = torch.tensor(inputs["attention_mask"])
        return inputs

    train_set = list(map(tokenizer_wrapper, dataset[:10]))
    test_set = list(map(tokenizer_wrapper, dataset[10:15]))
    val_set = list(map(tokenizer_wrapper, dataset[15:]))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, generate_labels=generate_labels))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, generate_labels=generate_labels))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, generate_labels=generate_labels))

    return train_loader, test_loader, val_loader
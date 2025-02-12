import torch
import torch.nn.functional as F

def custom_collate_fn(batch, tokenizer):
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

    # Pad to the max length item for the entire batch.
    # For inputs, use the padding token for padding. For the attention mask, pad with 0s.
    max_length = max([len(elem["input_ids"]) for elem in batch])
    input_ids = torch.stack([F.pad(elem["input_ids"], 
                            (0, max_length - len(elem["input_ids"])),
                            mode="constant",
                            value=tokenizer.pad_token_id) for elem in batch], dim=0)
    attention_mask = torch.stack([F.pad(elem["attention_mask"],
                                 (0, max_length - len(elem["attention_mask"])),
                                 mode="constant",
                                 value=0) for elem in batch], dim=0)

    attention_mask = attention_mask[:, 1:]
    pad_mask = torch.zeros(len(batch), 1)
    attention_mask = torch.cat((attention_mask, pad_mask), dim=1)

    pad_tokens = torch.full((len(batch), 1), tokenizer.pad_token_id)
    labels = input_ids.clone()[:, 1:] # Remove first token from labels.
    labels = torch.cat((labels, pad_tokens), dim=1) # Add another padding token
    labels = labels.masked_fill(attention_mask == 0, -100)

    assert labels.shape == input_ids.shape
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
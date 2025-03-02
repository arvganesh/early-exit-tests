import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

def custom_collate_fn(batch, tokenizer, generate_labels=True, nice_shape=True):
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
    if nice_shape:
        batch_length = 2 ** math.ceil(math.log(batch_length, 2))
    
    input_ids, attention_mask, loss_mask, labels = None, None, None, None
    if "input_ids" in batch[0]:
        input_ids = torch.stack([F.pad(elem["input_ids"], 
            (0, batch_length - len(elem["input_ids"])),
            mode="constant",
            value=tokenizer.pad_token_id) for elem in batch], dim=0)
    
    if "attention_mask" in batch[0]:
        attention_mask = torch.stack([F.pad(elem["attention_mask"],
            (0, batch_length - len(elem["attention_mask"])),
            mode="constant",
            value=0) for elem in batch], dim=0)

        attention_mask = attention_mask[:, 1:]
        pad_mask = torch.zeros(len(batch), 1)
        attention_mask = torch.cat((attention_mask, pad_mask), dim=1)

    if "loss_mask" in batch[0]:
        loss_mask = torch.stack([F.pad(elem["loss_mask"],
            (0, batch_length - len(elem["loss_mask"])),
            mode="constant",
            value=0) for elem in batch], dim=0)

        loss_mask = loss_mask[:, 1:]
        pad_mask = torch.zeros(len(batch), 1)
        loss_mask = torch.cat((loss_mask, pad_mask), dim=1)

    if generate_labels:
        pad_tokens = torch.full((len(batch), 1), tokenizer.pad_token_id)
        labels = input_ids.clone()[:, 1:] # Remove first token from labels.
        labels = torch.cat((labels, pad_tokens), dim=1) # Add another padding token
        labels = labels.masked_fill(attention_mask == 0, -100)
        assert labels.shape == input_ids.shape

    return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "labels": labels
            }

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

def get_toy_dataloaders(batch_size, tokenizer, max_length, generate_labels = True, nice_shape = True):
    dataset = [
            {
                "conversations": {
                    "from": ["human", "gpt"],
                    "value": [
                        "Hello, how are you?",
                        "I'm doing well, thank you!"
                        ]
                    }
                },
            {
                "conversations": {
                    "from": ["human", "gpt"],
                    "value": [
                        "What is the capital of France?",
                        "The capital of France is Paris."
                        ]
                    }
                },
            {
                "conversations": {
                    "from": ["human", "gpt"],
                    "value": [
                        "Tell me a joke.",
                        "Why did the chicken cross the road? To get to the other side!"
                        ]
                    }
                },
            {
                "conversations": {
                    "from": ["human", "gpt"],
                    "value": [
                        "What's the weather like today?",
                        "I'm not sure about the weather, but it's always sunny in the digital world."
                        ]
                    }
                },
            {
                "conversations": {
                    "from": ["human", "gpt"],
                    "value": [
                        "How do I bake a cake?",
                        "You can start by preheating your oven and mixing your ingredients carefully."
                        ]
                    }
                },
            {
                "conversations": {
                    "from": ["human", "gpt"],
                    "value": [
                        "What is 2+2?",
                        "2+2 equals 4."
                        ]
                    }
                },
        {
                "conversations": {
                    "from": ["human", "gpt"],
                    "value": [
                        "Can you write a poem?",
                        "Roses are red, violets are blue, I'm here to assist, just for you."
                        ]
                    }
                },
        {
                "conversations": {
                    "from": ["human", "gpt"],
                    "value": [
                        "What's the meaning of life?",
                        "The meaning of life is subjective, but many say it's to seek happiness and knowledge."
                        ]
                    }
                },
        {
                "conversations": {
                    "from": ["human", "gpt"],
                    "value": [
                        "Do you know any programming languages?",
                        "Yes, I can help you with Python, JavaScript, and several others."
                        ]
                    }
                },
        {
                "conversations": {
                    "from": ["human", "gpt"],
                    "value": [
                        "What is the tallest mountain?",
                        "Mount Everest is the tallest mountain in the world."
                        ]
                    }
                },
        {
                "conversations": {
                    "from": ["human", "gpt"],
                    "value": [
                        "Who wrote '1984'?",
                        "George Orwell is the author of '1984'."
                        ]
                    }
                },
        {
                "conversations": {
                    "from": ["human", "gpt"],
                    "value": [
                        "Can you help me with my homework?",
                        "Of course! What subject do you need help with?"
                        ]
                    }
                },
        {
                "conversations": {
                    "from": ["human", "gpt"],
                    "value": [
                        "What time is it?",
                        "I don't have a clock, but you can check your local time."
                        ]
                    }
                },
        {
                "conversations": {
                    "from": ["human", "gpt"],
                    "value": [
                        "What is the square root of 16?",
                        "The square root of 16 is 4."
                        ]
                    }
                },
        {
                "conversations": {
                    "from": ["human", "gpt"],
                    "value": [
                        "Tell me something interesting.",
                        "Did you know that honey never spoils?"
                        ]
                    }
                },
        {
                "conversations": {
                    "from": ["human", "gpt"],
                    "value": [
                        "Can you translate 'hello' into Spanish?",
                        "Sure, 'hello' in Spanish is 'hola'."
                        ]
                    }
                },
        {
                "conversations": {
                    "from": ["human", "gpt"],
                    "value": [
                        "What's your favorite color?",
                        "I don't have preferences, but blue is often a popular choice."
                        ]
                    }
                },
        {
                "conversations": {
                    "from": ["human", "gpt"],
                    "value": [
                        "How do I solve a quadratic equation?",
                        "You can use the quadratic formula: (-b ± √(b² - 4ac))/(2a)."
                        ]
                    }
                },
        {
                "conversations": {
                    "from": ["human", "gpt"],
                    "value": [
                        "What is AI?",
                        "AI stands for artificial intelligence, which is the simulation of human intelligence in machines."
                        ]
                    }
                },
        {
                "conversations": {
                    "from": ["human", "gpt"],
                    "value": [
                        "Tell me a fun fact.",
                        "The heart of a shrimp is located in its head."
                        ]
                    }
                }
    ]

    def tokenizer_wrapper(example):
        return tokenize_sharegpt_examples(example, tokenizer, max_length)

    def convert_to_tensor(inputs):
        return {k: torch.tensor(v) for k, v in inputs.items()}


    train_set = list(map(tokenizer_wrapper, dataset[:10]))
    test_set = list(map(tokenizer_wrapper, dataset[10:15]))
    val_set = list(map(tokenizer_wrapper, dataset[15:]))

    train_set = list(map(convert_to_tensor, train_set))
    test_set = list(map(convert_to_tensor, test_set))
    val_set = list(map(convert_to_tensor, val_set))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, generate_labels=generate_labels, nice_shape=nice_shape))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, generate_labels=generate_labels, nice_shape=nice_shape))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, generate_labels=generate_labels, nice_shape=nice_shape))

    return train_loader, test_loader, val_loader

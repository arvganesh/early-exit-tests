from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

import torch
import numpy as np
import argparse
import random
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--compile", action='store_true')
parser.add_argument("-d", "--dynamic", action='store_true')
parser.add_argument("-s", "--static_shape", action='store_true')
parser.add_argument("-r", "--round_to_power_of_two", action='store_true')
parser.add_argument("-n", "--num_tokens", type=int, required=True)
args = parser.parse_args()

if args.round_to_power_of_two:
    args.num_tokens = 2 ** math.ceil(math.log2(args.num_tokens))

# Make sure results are deterministic.
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True

model_path = "meta-llama/llama-3.2-1B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get input tokens.
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
truth_model = AutoModelForCausalLM.from_pretrained(model_path)
truth_model = truth_model.to(device)
if args.compile:
    truth_model = torch.compile(truth_model, mode="max-autotune", dynamic=args.dynamic)
num_original_layers = len(truth_model.model.layers) - 1
prompt = "Hello!"
if args.static_shape:
    inputs = tokenizer(prompt, padding="max_length", max_length=args.num_tokens, return_tensors="pt")
else:
    inputs = tokenizer(prompt, return_tensors="pt")
input_toks = inputs["input_ids"]
attn_mask = inputs["attention_mask"].to(device)
num_prompt_tok = inputs["attention_mask"].sum().item()
num_additional_tokens = args.num_tokens - num_prompt_tok
input_toks = input_toks.to(device)

# Run experiments.
with torch.no_grad():
    total_time = 0
    data = []
    pprint(args)
    for i in range(num_additional_tokens):
        # Do forward pass.
        start = time.time()
        outputs = truth_model(input_toks, attention_mask=attn_mask)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time() 
        data.append(end - start)
        total_time += end - start
        
        # Sample next token.
        logits = outputs.logits
        new_token_index = num_prompt_tok - 1 + i if args.static_shape else -1
        probs = torch.softmax(logits[:, new_token_index, :], dim=-1) # 1 x 1 x Vocab Size
        new_token = torch.multinomial(probs, num_samples=1) # 1 x 1
        if args.static_shape:
            input_toks[:, num_prompt_tok + i] = new_token[0, 0]
            attn_mask[:, num_prompt_tok + i] = 1
        else:
            input_toks = torch.cat((input_toks, new_token), dim=-1)
            attn_mask = torch.cat((attn_mask, torch.ones((1, 1), device=device)), dim=-1)
        
    
    data = torch.tensor(data)
    median = torch.median(data).item()
    mean = torch.mean(data).item()
    print(f"Average: {mean:.5f} seconds per token")
    print(f"Min: {min(data):.5f}, Max: {max(data):.5f}, Median: {median:.5f}")
    print(tokenizer.decode(input_toks[0], skip_special_tokens=True))
    print(data)

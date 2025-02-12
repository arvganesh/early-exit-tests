from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import torch
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Make sure results are deterministic.
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

model_path = "meta-llama/llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
truth_model = AutoModelForCausalLM.from_pretrained(model_path)
num_original_layers = len(truth_model.model.layers) - 1
prompt = "Hello!"
inputs = tokenizer(prompt, return_tensors="pt")

"""
tensor([[[ 0.0028,  0.0033, -0.0099,  ..., -0.0018,  0.0008,  0.0007],
         [ 0.0008,  0.0060,  0.0194,  ...,  0.0211,  0.0166, -0.0127],
         [ 0.0045,  0.0166,  0.0210,  ..., -0.0054, -0.0422, -0.0315]]])
"""
outputs = truth_model(inputs["input_ids"], output_hidden_states=True)

import pdb; pdb.set_trace()
# 
# range of histogram
plt.hist(data_sizes, bins=10)
plt.xlim(0, 1000)
plt.savefig("data_size_histogram.png")

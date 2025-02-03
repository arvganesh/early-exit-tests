from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

train_dataset = load_dataset("DKYoon/SlimPajama-6B", split="train", num_proc=8)
validation_dataset = load_dataset("DKYoon/SlimPajama-6B", split="validation", num_proc=8)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=16, shuffle=True)

import code; code.interact(local=dict(globals(), **locals()))

# write code to create a histogram of the data_sizes in train_loader
# data_sizes = [len(d[]) for d in train_loader]
# plt.hist(data_sizes, bins=10)
# plt.savefig("data_size_histogram.png")
# 
# range of histogram
plt.hist(data_sizes, bins=10)
plt.xlim(0, 1000)
plt.savefig("data_size_histogram.png")

import torch
import torch.nn.functional as F
import argparse
import random
import numpy
from truncated_llama import TruncatedLlama
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup
from share_gpt_dataset import get_sharegpt_dataloaders
from data_utils import get_toy_dataloaders

parser = argparse.ArgumentParser(description="Evaluate the perplexity of trained vs. baseline models.")
parser = argparse.ArgumentParser(description="Train a truncated Llama model with a tuned head.")
parser.add_argument(
    "--model_path",
    type=str,
    default="meta-llama/Llama-3.2-1B",
    help="Path to the model checkpoint or model identifier."
)
parser.add_argument(
    "--target_layer",
    type=int,
    default=15,
    help="The transformer layer number that you want to target."
)
parser.add_argument(
    "--weights_to_load",
    type=str,
    default=None,
    help="Path of weights to load into the lm head."
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Name of back-end device to use for PyTorch."
)
parser.add_argument(
    "--seed",
    type=int,
    default=0
)
args = parser.parse_args()

torch.manual_seed(args.seed)
numpy.random.seed(args.seed)
random.seed(args.seed)
#torch.use_deterministic_algorithms(True)

torch.set_float32_matmul_precision("high")
model = TruncatedLlama(args.model_path, 
                       early_exit_idx=args.target_layer,
                       lm_head_random_init=False)

model_dict = torch.load(args.weights_to_load, map_location=args.device)
model.load_from_checkpoint(model_dict["lm_head"], model_dict["last_transformer"])
model.to(args.device)

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.pad_token = tokenizer.eos_token

llama_model = AutoModelForCausalLM.from_pretrained(args.model_path)
llama_model.to(args.device)
llama_model.eval()

total = sum(p.numel() for p in llama_model.parameters())
print(f"total # in base llama: {total}") 

prompt = "Hi! I'm a language model."
inputs = tokenizer(prompt, return_tensors="pt")
og_input_ids = inputs["input_ids"]
og_input_ids = og_input_ids.to(args.device)

inputs = tokenizer(prompt, return_tensors="pt")
actual_input_ids = inputs["input_ids"]
actual_input_ids = actual_input_ids.to(args.device)

my_outputs = model.generate(actual_input_ids, 20, tokenizer.eos_token_id)
llama_outputs = llama_model.generate(input_ids=og_input_ids, max_new_tokens=20)

my_text = tokenizer.decode(my_outputs[0], skip_special_tokens=True)
llama_text = tokenizer.decode(llama_outputs[0], skip_special_tokens=True)

print(my_text)
print(" ------ ")
print(llama_text)


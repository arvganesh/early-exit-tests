import torch
import torch.nn.functional as F
import argparse
import random
import numpy
import time
from truncated_llama import TruncatedLlama
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup
from share_gpt_dataset import get_sharegpt_dataloaders
from data_utils import get_toy_dataloaders

parser = argparse.ArgumentParser(description="Evaluate the average agreement length between baseline and tuned models.")
parser.add_argument(
    "--target_model_path",
    type=str,
    default="meta-llama/Llama-2-7b-chat-hf",
    help="Path to the model checkpoint or model identifier."
)
parser.add_argument(
    "--draft_model_path",
    type=str,
    default="meta-llama/Llama-2-7b-chat-hf",
    help="Path to the model checkpoint or model identifier."
)
parser.add_argument(
    "--softmax_temperature",
    type=float,
    default=1.0,
    help="Temperature value for softmax."
)
parser.add_argument(
    "--max_length",
    type=int,
    default=4096,
    help="Maximum sequence length for the tokenizer."
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
parser.add_argument(
    "--toy_dataset",
    action="store_true"
)
#parser.add_argument(
#    "--flash_attn",
#    action="store_true"
#)   
parser.add_argument(
    "--vanilla_decode",
    action="store_true"
)
parser.add_argument(
    "--speculate_len",
    type=int
)
parser.add_argument(
    "--num_iters",
    type=int
)
parser.add_argument(
    "--num_vanilla_tok",
    type=int
)
args = parser.parse_args()

def set_seeds():
    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)
    random.seed(args.seed)
#torch.use_deterministic_algorithms(True)

torch.set_float32_matmul_precision("high")

target_model = AutoModelForCausalLM.from_pretrained(args.target_model_path)
draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model_path)

target_model.eval()
draft_model.eval()
target_model.to(args.device)
draft_model.to(args.device)

tokenizer = AutoTokenizer.from_pretrained(args.draft_model_path)
tokenizer.pad_token = tokenizer.eos_token

# Temp = 0 -> argmax.
def temperature_softmax(logits, temp=1.0):
    logits /= temp
    return torch.softmax(logits, dim=-1)

# evaluate average perplexity over test dataset
# Get data tensors, move to device.
with torch.no_grad():
    prompt = "Hi! I'm a language model."
    tokenizer_out = tokenizer(prompt, return_tensors="pt")
    input_ids = tokenizer_out["input_ids"].to(args.device)
    
    if args.vanilla_decode:
        start = time.time()
        for _ in range(args.num_vanilla_tok):
            target_out = target_model(input_ids)
            target_logits = target_out.logits[:, -1, :]
            target_probs = temperature_softmax(target_logits, temp=args.softmax_temperature)
            sample = torch.multinomial(target_probs, 1)
            input_ids = torch.cat((input_ids, sample), dim=-1)
        torch.cuda.synchronize()
        end = time.time()
        print(" --- Vanilla Decoding ---- ")
        print(f"Tokens per second: { args.num_vanilla_tok / (end - start):.4f}")
        print(tokenizer.decode(input_ids[0], skip_special_tokens=True))   
    else:
        num_tokens_generated = 0
        start = time.time()
        for num_iters in range(args.num_iters):
            # Get draft tokens
            input_length = input_ids.size(-1)
            for i in range(args.speculate_len):
                draft_out = draft_model(input_ids) 
                logits = draft_out.logits[:, -1, :]
                probs = temperature_softmax(logits, temp=args.softmax_temperature)
                sample = torch.multinomial(probs, 1)
                input_ids = torch.cat((input_ids, sample), dim=-1)
           
            # Verify with target.
            target_out = target_model(input_ids)
            target_probs = temperature_softmax(target_out.logits, temp=args.softmax_temperature)
            draft_probs = temperature_softmax(draft_out.logits, temp=args.softmax_temperature) 

            # Find disagreement point.
            agree_length = 0
            randoms = torch.rand(args.speculate_len)
            for i in range(args.speculate_len):
                sample_idx = input_ids[0, input_length + i].cpu().item()
                likelihood = target_probs[:, input_length + i - 1, sample_idx].item() / draft_probs[:, input_length + i - 1, sample_idx].item()
                if likelihood < randoms[i]:
                    break
                agree_length += 1
                
            # Fix the disagreed upon sample if needed.
            disagree_idx = input_length + agree_length 
            if agree_length < args.speculate_len:
                target_dist = target_probs[:, disagree_idx - 1]
                draft_dist = draft_probs[:, disagree_idx - 1]
                fix_dist = torch.clamp(target_dist - draft_dist, min=0)
                fix_dist /= fix_dist.sum()
                new_token = torch.multinomial(fix_dist, 1)
                input_ids[:, disagree_idx] = new_token
                input_ids = input_ids[:, :disagree_idx + 1]
            else:
                new_token = torch.multinomial(target_probs[:, -1, :], 1)
                input_ids = torch.cat((input_ids, new_token), dim=-1)

            num_tokens_generated += agree_length + 1
            
        torch.cuda.synchronize()
        end = time.time()
        generation_time = end - start
        print(" --- Speculative Decoding ---- ")
        print(f"Num Tokens Generated: { num_tokens_generated }")
        print(f"Tokens per second: { num_tokens_generated / generation_time:.4f}")
        print(tokenizer.decode(input_ids[0], skip_special_tokens=True))    

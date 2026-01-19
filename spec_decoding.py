import torch
import torch.nn.functional as F
import argparse
import random
import numpy
import time
from truncated_llama import TruncatedLlama
from sampling_utils import sampling_probs_from_logits, safe_normalize
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    "--top_p",
    type=float,
    default=1.0,
    help="Nucleus sampling p (1.0 disables).",
)
parser.add_argument(
    "--top_k",
    type=int,
    default=0,
    help="Top-k sampling (0 disables).",
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
use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()

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

def _safe_normalize(dist: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return safe_normalize(dist, eps=eps)

# evaluate average perplexity over test dataset
# Get data tensors, move to device.
with torch.no_grad():
    prompt = "Hi! I'm a language model."
    tokenizer_out = tokenizer(prompt, return_tensors="pt")
    prompt_ids = tokenizer_out["input_ids"]
    prompt_len = prompt_ids.size(-1) 
    if args.vanilla_decode:
        input_ids = prompt_ids.to(args.device)
        start = time.time()
        for _ in range(args.num_vanilla_tok):
            target_out = target_model(input_ids)
            target_logits = target_out.logits[:, -1, :]
            target_probs = sampling_probs_from_logits(
                target_logits,
                temperature=args.softmax_temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            sample = torch.multinomial(target_probs, 1)
            input_ids = torch.cat((input_ids, sample), dim=-1)
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print(" --- Vanilla Decoding ---- ")
        print(f"Tokens per second: { args.num_vanilla_tok / (end - start):.4f}")
        print(tokenizer.decode(input_ids[0], skip_special_tokens=True))   
    else:
        num_tokens_generated = 0
        start = time.time()
        
        max_token_amt = prompt_len + args.num_iters * (args.speculate_len + 1)
        input_ids = torch.zeros((1, max_token_amt), dtype=prompt_ids.dtype).to(args.device)
        attn_mask = torch.zeros((1, max_token_amt)).to(args.device) 
        attn_mask[:, :prompt_len] = 1
        input_ids[:, :prompt_len] = prompt_ids[:, :]

        for num_iters in range(args.num_iters):
            # Get draft tokens
            input_length = prompt_len + num_tokens_generated
            draft_logits_list = []
            draft_tokens_list = []
            for i in range(args.speculate_len):
                draft_out = draft_model(input_ids, attention_mask=attn_mask)
                logits = draft_out.logits[:, input_length + i - 1, :]  # predicts token at position input_length + i
                probs = sampling_probs_from_logits(
                    logits,
                    temperature=args.softmax_temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                sample = torch.multinomial(probs, 1)
                draft_logits_list.append(logits)
                draft_tokens_list.append(sample)
                input_ids[:, input_length + i] = sample.item()
                attn_mask[:, input_length + i] = 1
           
            # Verify with target.
            target_out = target_model(input_ids, attention_mask=attn_mask)
            target_probs = sampling_probs_from_logits(
                target_out.logits,
                temperature=args.softmax_temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            overlap_sum = 0.0
            
            # Find disagreement point.
            agree_length = 0
            for i in range(args.speculate_len):
                pred_idx = input_length + i - 1
                tok = draft_tokens_list[i]  # (1,1)
                p = target_probs[:, pred_idx, :]  # (1,V)
                q = sampling_probs_from_logits(
                    draft_logits_list[i],
                    temperature=args.softmax_temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                overlap_sum += torch.minimum(p, q).sum().item()
                p_tok = p.gather(-1, tok)
                q_tok = q.gather(-1, tok).clamp_min(1e-12)
                accept_prob = torch.minimum(torch.ones_like(p_tok), p_tok / q_tok)
                u = torch.rand_like(accept_prob)
                if (u < accept_prob).all():
                    agree_length += 1
                    continue
                break
               
            # Fix the disagreed upon sample if needed.
            disagree_idx = input_length + agree_length 
            if agree_length < args.speculate_len:
                pred_idx = disagree_idx - 1
                target_dist = target_probs[:, pred_idx, :]
                draft_dist = sampling_probs_from_logits(
                    draft_logits_list[agree_length],
                    temperature=args.softmax_temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                fix_dist = torch.clamp(target_dist - draft_dist, min=0.0)
                fix_dist = _safe_normalize(fix_dist)
                if not torch.isfinite(fix_dist).all() or fix_dist.sum().item() == 0.0:
                    fix_dist = target_dist
                new_token = torch.multinomial(fix_dist, 1)
            else:
                new_token = torch.multinomial(target_probs[:, -1, :], 1)

            input_ids[:, disagree_idx] = new_token
            num_tokens_generated += agree_length + 1
            if args.speculate_len > 0:
                print(f"iter={num_iters} accepted={agree_length}/{args.speculate_len} overlap={overlap_sum / args.speculate_len:.4f}")
            
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        generation_time = end - start
        print(" --- Speculative Decoding ---- ")
        print(f"Num Tokens Generated: { num_tokens_generated }")
        print(f"Tokens per second: { num_tokens_generated / generation_time:.4f}")
        print(tokenizer.decode(input_ids[0], skip_special_tokens=True))    

import torch
import os
import glob
import torch.nn.functional as F
import argparse
import json
import random
import statistics
import numpy as np
from truncated_llama import TruncatedLlama
from sampling_utils import sampling_probs_from_logits, safe_normalize
from transformers import AutoTokenizer
from share_gpt_dataset import get_sharegpt_dataloaders
from data_utils import get_toy_dataloaders
from fineweb_dataset import get_fineweb_dataloaders

parser = argparse.ArgumentParser(description="Evaluate the average agreement length between baseline and tuned models.")
parser.add_argument(
    "--model_path",
    type=str,
    default="meta-llama/Llama-2-7b-chat-hf",
    help="Path to the model checkpoint or model identifier."
)
parser.add_argument(
    "--target_layer",
    type=int,
    default=15,
    help="The transformer layer number that you want to target."
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
    "--run_type",
    type=str,
    help=""
)
parser.add_argument(
    "--sample_from_models",
    action="store_true",
    help="Sample from baseline and experimental models and print the results."
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
parser.add_argument(
    "--flash_attn",
    action="store_true"
)   
parser.add_argument(
    "--speculate_len",
    type=int,
    default=8,
    help="Number of draft tokens proposed per target verification (gamma).",
)
parser.add_argument(
    "--num_iters",
    type=int,
    default=256,
    help="Number of target verification iterations per example.",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=500,
    help="Number of dataset examples to evaluate.",
)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
#torch.use_deterministic_algorithms(True)

torch.set_float32_matmul_precision("high")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.pad_token = tokenizer.eos_token

# Temp = 0 -> argmax.
def temperature_softmax(logits, temp=1.0):
    logits = logits / temp
    return torch.softmax(logits, dim=-1, dtype=torch.float32)

def _safe_normalize(dist: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return safe_normalize(dist, eps=eps)

def spec_step(
    model: TruncatedLlama,
    input_ids: torch.Tensor,
    temperature: float,
    speculate_len: int,
    top_p: float,
    top_k: int,
) -> tuple[torch.Tensor, int, float]:
    """
    Runs one speculative decoding iteration:
      - propose `speculate_len` draft tokens from the early-exit model
      - verify with target (reference) model
      - accept/reject per the exact sampling-correct rule

    Returns:
      updated_input_ids, num_accepted_draft_tokens, mean_per_token_overlap_for_this_block
    """
    device = input_ids.device
    attn_mask = torch.ones_like(input_ids, device=device)
    prefix_len = input_ids.size(1)

    draft_logits_list = []
    draft_tokens_list = []

    # Draft proposes gamma tokens sequentially.
    for _ in range(speculate_len):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            draft_out = model.truncated_model(input_ids, attention_mask=attn_mask)
        logits = draft_out.logits[:, -1, :]  # predicts next token
        q = sampling_probs_from_logits(logits, temperature=temperature, top_p=top_p, top_k=top_k)  # (B, V)
        tok = torch.multinomial(q, 1)  # (B, 1)
        draft_logits_list.append(logits)
        draft_tokens_list.append(tok)
        input_ids = torch.cat([input_ids, tok], dim=1)
        attn_mask = torch.cat([attn_mask, torch.ones((attn_mask.size(0), 1), device=device, dtype=attn_mask.dtype)], dim=1)

    # Verify with target once on the full proposed prefix (causal => logits don't depend on future tokens).
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            ref_out = model.reference_model(input_ids, attention_mask=attn_mask)
    ref_logits = ref_out.logits  # (B, T, V)

    num_accepted = 0
    overlaps = []

    # Accept/reject sequentially.
    for j in range(speculate_len):
        tok = draft_tokens_list[j]  # (B, 1)

        # Distributions for token at position prefix_len + j are predicted at index (prefix_len + j - 1).
        pred_idx = prefix_len + j - 1

        p = sampling_probs_from_logits(ref_logits[:, pred_idx, :], temperature=temperature, top_p=top_p, top_k=top_k)  # (B, V)
        q = sampling_probs_from_logits(draft_logits_list[j], temperature=temperature, top_p=top_p, top_k=top_k)        # (B, V)
        overlaps.append(torch.minimum(p, q).sum(dim=-1))  # per-batch

        p_tok = p.gather(-1, tok)  # (B, 1)
        q_tok = q.gather(-1, tok).clamp_min(1e-12)
        accept_prob = torch.minimum(torch.ones_like(p_tok), p_tok / q_tok)  # (B, 1)
        u = torch.rand_like(accept_prob)

        if (u < accept_prob).all():
            num_accepted += 1
            continue

        # Rejection at token j: sample from corrected distribution proportional to (p - q)+.
        fix = torch.clamp(p - q, min=0.0)
        fix = _safe_normalize(fix)
        if not torch.isfinite(fix).all() or (fix.sum(dim=-1) == 0).any():
            fix = p
        new_tok = torch.multinomial(fix, 1)

        # Keep accepted tokens, replace rejected one, and discard remaining proposals.
        input_ids = input_ids[:, : prefix_len + j]  # accepted prefix (prompt + accepted draft tokens)
        input_ids = torch.cat([input_ids, new_tok], dim=1)
        mean_overlap = torch.stack(overlaps, dim=0).mean().item() if overlaps else 0.0
        return input_ids, num_accepted, mean_overlap

    # Accepted all gamma draft tokens: sample one token from target at the end.
    p_next = sampling_probs_from_logits(ref_logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
    next_tok = torch.multinomial(p_next, 1)
    input_ids = torch.cat([input_ids, next_tok], dim=1)

    mean_overlap = torch.stack(overlaps, dim=0).mean().item() if overlaps else 0.0
    return input_ids, num_accepted, mean_overlap

#temps = [0.0, 0.2, 1.0]
temps = [args.softmax_temperature]

scratch = "/scratch/10543/arvganesh/"
#run_type = "only_head"
model_name = args.model_path.split("/")[-1]
base_path = f"/scratch/10543/arvganesh/models/{model_name}/{args.run_type}/"
models = os.listdir(base_path)

def find_model_path(layer_idx):
    for model in models:
        if model.startswith(f"layer{layer_idx}"):
            model_dir = os.path.join(base_path, model)
            checkpoints = os.listdir(model_dir) 
            checkpoint = [x for x in checkpoints if x.startswith("model_80000")][0] 
            return str(os.path.join(model, checkpoint))
    
    assert False

model_paths = []
#model_paths.append(find_model_path(2 * args.target_layer))     
#model_paths.append(find_model_path(2 * args.target_layer + 1))     
model_paths.append(find_model_path(args.target_layer))

model_paths = []

for run in run_names:
  folder = os.path.join(base_path, run, "model_15000*")
  matches = glob.glob(folder)
  if matches:
    model_paths.append(matches[0])
  else:
    assert False

assert(len(model_paths) == len(run_names))

print(model_paths)

agreement_stats_all = {}

print(model_paths)

# evaluate average perplexity over test dataset
# Get data tensors, move to device.
for path in model_paths:
    print(f"Evaluating {path}")
    # path: layerX_Ysteps_beginTIMESTAMP/model_NSTEPS_loss.pt
    folder_name = path.split("/")[0]
    layer_name = folder_name.split("_")[0]
    layer_idx = int(layer_name[len("layer"):])
    model = TruncatedLlama(args.model_path, 
                           early_exit_idx=layer_idx,
                       lm_head_random_init=False,
                       use_flash_attn=args.flash_attn)
    model_dict = torch.load(os.path.join(base_path, path), map_location=args.device)
    model.new_lm_head.load_state_dict(model_dict["lm_head"])
    if model_dict["last_transformer"]:
        print("loading last transformer layer!")
        model.headless_model.layers[layer_idx].load_state_dict(model_dict["last_transformer"])
    model.eval()
    model.to(args.device)
    
    agreement_stats_mdl = {}

    for temp in temps:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        print(f"Evaluating at temp={temp}")
        with torch.no_grad():
            if args.toy_dataset:
                train, test, val = get_toy_dataloaders(1, tokenizer, args.max_length, generate_labels = True, nice_shape = False)
            else:
                #train, test, val = get_sharegpt_dataloaders(1, tokenizer, args.max_length, generate_labels = True, nice_shape = False)
                train, test, val = get_fineweb_dataloaders(
                    1, # batch size
                    tokenizer,
                    args.max_length,
                    False, # generate labels
                    seed = args.seed
                )

            accepted_stats = []
            overlap_stats = []
            print(f"Num examples: {len(val)}")
            for idx, batch in enumerate(val):
                if idx >= args.num_samples:
                    break 
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                true_len = int(attention_mask.sum().item())
                prompt_len = max(1, true_len // 2)
                input_ids = input_ids[:, :prompt_len]

                total_accepted = 0
                total_overlap = 0.0
                for _ in range(args.num_iters):
                    input_ids, accepted, mean_overlap = spec_step(
                        model=model,
                        input_ids=input_ids,
                        temperature=temp,
                        speculate_len=args.speculate_len,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                    total_accepted += accepted
                    total_overlap += mean_overlap

                accepted_stats.append(total_accepted / args.num_iters)
                overlap_stats.append(total_overlap / args.num_iters)
        
        avg_accepted = float(np.mean(accepted_stats)) if accepted_stats else 0.0
        avg_overlap = float(np.mean(overlap_stats)) if overlap_stats else 0.0
        avg_tv = 1.0 - avg_overlap
        min_accepted = min(accepted_stats) if accepted_stats else 0.0
        max_accepted = max(accepted_stats) if accepted_stats else 0.0
        median_accepted = statistics.median(accepted_stats) if accepted_stats else 0.0
        std_accepted = statistics.stdev(accepted_stats) if len(accepted_stats) > 1 else 0.0
        percentiles = np.percentile(accepted_stats, [25, 50, 75]) if accepted_stats else [0.0, 0.0, 0.0]

        print(f"Avg accepted draft tokens per target call: {avg_accepted:.5f} (gamma={args.speculate_len})")
        print(f"Avg per-token overlap (sum(min(p,q))): {avg_overlap:.5f} (TV={avg_tv:.5f})")
        agreement_stats_mdl[temp] = {
            "accepted_per_iter": accepted_stats,
            "overlap_per_iter": overlap_stats,
            "avg_accepted": avg_accepted,
            "avg_overlap": avg_overlap,
            "avg_tv": avg_tv,
            "min_accepted": min_accepted,
            "max_accepted": max_accepted,
            "median_accepted": median_accepted,
            "std_accepted": std_accepted,
            "percentiles255075": list(percentiles),
            "speculate_len": args.speculate_len,
            "num_iters": args.num_iters,
            "num_samples": args.num_samples,
            "top_p": args.top_p,
            "top_k": args.top_k,
        }
        agreement_stats_all[path] = agreement_stats_mdl
    
    
    # Write the per-layer agreement stats into a single JSON file
    save_dir = os.path.join(scratch, f"evaluation/{args.run_type}/")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"agreement_stats_by_layer{layer_idx}_{temps[0]}.json")
    with open(save_path, "w") as outfile:
        json.dump(agreement_stats_all, outfile, indent=4)

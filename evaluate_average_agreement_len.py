import torch
import os
import torch.nn.functional as F
import argparse
import json
import random
import statistics
import numpy as np
from truncated_llama import TruncatedLlama
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup
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
    if temp == 0:
        max_idx = torch.argmax(logits)
        onehot = F.one_hot(max_idx, num_classes=logits.size(-1)).to(torch.float32)
        return torch.unsqueeze(onehot, 0)
    logits /= temp
    return torch.softmax(logits, dim=-1)

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

            total_agreement_tok = 0
            agreement_stats = []
            print(f"Num examples: {len(val)}")
            num_samples = 5000
            for idx, batch in enumerate(val):
                if idx >= num_samples:
                    break 
                input_ids = batch["input_ids"]
                input_ids = input_ids[:, :input_ids.size(1) // 2]
                input_ids = input_ids.to(args.device)
                agreement_length = 0
                
                should_print = False
                if should_print:
                    print(f"{(idx + 1) / len(val) * 100:.2f}% of the way there!")
                    print(f"Initial prompt: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")

                for i in range(2048):
                    with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
                        outputs = model(input_ids, keep_og_logits=True)

                    # Early exit with finetuned head.  
                    logits = outputs["logits"][:, -1, :] 
                    ee_probs = temperature_softmax(logits, temp=temp)
                    ee_sample = torch.multinomial(ee_probs, 1) 
                    
                    # Baseline Llama (unmodified)
                    og_logits = outputs["og_lm_logits"][:, -1, :]
                    og_probs = temperature_softmax(og_logits, temp=temp) 
                    og_sample = torch.multinomial(og_probs, 1)

                    """
                    For speculative decoding, there are two cases where disagreement can occur:
                    
                    1. Sampled tokens don't match.
                    2. Sampled tokens match, but we "oversampled" in the draft model and reject the original token. 
                    After resampling from an adjusted distribution, we might disagree.
                    """
                    if not torch.equal(ee_sample, og_sample):
                        if should_print:
                            print(f"ee_tok: {ee_sample.item()} og_tok: {og_sample.item()}")
                            print(f"P({ee_sample.item()} in EE_MODEL) = {ee_probs[0, ee_sample.item()]}")
                            print(f"P({ee_sample.item()} in OG_MODEL) = {og_probs[0, ee_sample.item()]}")
                            print(f"P({og_sample.item()} in EE_MODEL) = {ee_probs[0, og_sample.item()]}")
                            print(f"P({og_sample.item()} in OG_MODEL) = {og_probs[0, og_sample.item()]}")
                        break
                    elif temp != 0:
                        # We oversampled from the draft model. 
                        sample = ee_sample[0].item()
                        if ee_probs[:, sample] > og_probs[:, sample]:
                            break
                    agreement_length += 1
                    input_ids = torch.cat((input_ids, ee_sample), dim=1)
                
                if should_print and agreement_length >= 10:
                    print(agreement_length, idx)
                agreement_stats.append(agreement_length)
                total_agreement_tok += agreement_length 
        
        avg_agreement = total_agreement_tok / num_samples
        min_agreement = min(agreement_stats)
        max_agreement = max(agreement_stats)
        median_agreement = statistics.median(agreement_stats)
        std_agreement = statistics.stdev(agreement_stats) if len(agreement_stats) > 1 else 0.0
        percentiles = np.percentile(agreement_stats, [25, 50, 75])
        print(f"Average agreement length: {total_agreement_tok / num_samples:.5f}")
        agreement_stats_mdl[temp] = {
            "data": agreement_stats,
            "avg_agreement": avg_agreement,
            "min_agreement": min_agreement,
            "max_agreement": max_agreement,
            "median_agreement": median_agreement,
            "std_agreement": std_agreement,
            "percentiles255075": list(percentiles)
        }
        agreement_stats_all[path] = agreement_stats_mdl
    
    
    # Write the per-layer agreement stats into a single JSON file
    save_dir = os.path.join(scratch, f"evaluation/{args.run_type}/")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"agreement_stats_by_layer{layer_idx}_{temps[0]}.json")
    with open(save_path, "w") as outfile:
        json.dump(agreement_stats_all, outfile, indent=4)

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
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup
from share_gpt_dataset import get_sharegpt_dataloaders
from data_utils import get_toy_dataloaders

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


temps = [args.softmax_temperature]

base_path = "/scratch/10543/arvganesh/models/Llama-3.2-1B-Instruct/"

run_names = [
#    "layer10_50000steps_begin1740943862",
#    "layer11_50000steps_begin1740943852",
#    "layer12_50000steps_begin1740943846",
#    "layer13_50000steps_begin1740943840",
    "layer14_50000steps_begin1740943832",
#    "layer1_50000steps_begin1740943912",
#    "layer2_50000steps_begin1740943909",
#    "layer3_50000steps_begin1740943905",
#    "layer4_50000steps_begin1740943903",
#    "layer5_50000steps_begin1740943899",
#    "layer6_50000steps_begin1740943884",
#    "layer7_50000steps_begin1740943877",
#    "layer8_50000steps_begin1740943873",
#    "layer9_50000steps_begin1740943862"
]

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

baseline_llama = AutoModelForCausalLM.from_pretrained(args.model_path)
baseline_llama.eval()
baseline_llama.to(args.device)

# evaluate average perplexity over test dataset
# Get data tensors, move to device.
for path in model_paths:
    print(f"Evaluating {path}")

    folder_name = path.split("/")[-2]
    layer_name = folder_name.split("_")[0]
    layer_idx = int(layer_name[len("layer"):])
    model = TruncatedLlama(args.model_path, 
                           early_exit_idx=layer_idx,
                           lm_head_random_init=False,
                           use_flash_attn=args.flash_attn,
                           inference_mode=True)
    model.new_lm_head.load_state_dict(torch.load(path, map_location=args.device))
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
                train, test, val = get_sharegpt_dataloaders(1, tokenizer, args.max_length, generate_labels = True, nice_shape = False)
            agreement_stats = []
            print(f"Num examples: {len(test)}")
            # todo modify to use inference mode
            # change agreement length calculation to match sujay's msg + speculative decoding paper.
            # generate tokens and verify w/ target.
            # # accepted tokens on avg is what we care about bc that's waht produces teh speedup
            for idx, batch in enumerate(test):
                input_tokens, loss_mask = batch["input_ids"], batch["loss_mask"]
                
                print(tokenizer.decode(input_tokens[0, :9], skip_special_tokens=True))
                should_print = True 
                if should_print:
                    print(f"{(idx + 1) / len(val) * 100:.2f}% of the way there!")
                    print(f"Initial prompt: {tokenizer.decode(input_tokens[0], skip_special_tokens=True)} \n")
                
                """
                input_tokens:
                 H  H  H  A  A  A
                [a, b, c, d, e, f]
                [0, 0, 1, 1, 1, 0]
                """
                example_length = input_tokens.size(-1)
                input_ids = torch.zeros(1, example_length, dtype=torch.int)
                attention_mask = torch.zeros(1, example_length, dtype=torch.int)
                input_ids, attention_mask = input_ids.to(args.device), attention_mask.to(args.device)
                i = 0 # where we are on the current training example
                while i < example_length - 1:
                    # Fill up context tokens.
                    while i < example_length - 1: 
                        input_ids[:, i] = input_tokens[:, i]
                        attention_mask[:, i] = 1
                        if loss_mask[:, i] == 1:
                            break
                        i += 1
                    
                    # Compute agreement of draft and target using the example as context.
                    # We are NOT looking at whether the draft model's output matches the dataset.
                    start_idx = i
                    draft_probs = []
                    while i < example_length - 1 and loss_mask[:, i] == 1:
                        with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
                            outputs = model(input_ids, attention_mask=attention_mask)

                        i += 1
                        logits = outputs["logits"][:, -1, :] 
                        probs = temperature_softmax(logits, temp=temp)
                        sample = torch.multinomial(probs, 1)
                        input_ids[:, i] = sample[0].item()
                        draft_probs.append(probs[:, input_ids[:, i]])
                        attention_mask[:, i] = 1
                    
                    print("During eval: ")
                    print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
                    print("\n")
                    # Verify with target model.
                    target_outputs = baseline_llama(input_ids, attention_mask=attention_mask)
                    target_logits = target_outputs["logits"]
                    
                    agree = True
                    num_tok_generated = 1 # start at 1 bc we always produce at least one
                    for j in range(len(draft_probs)):
                        draft_idx = start_idx + j + 1 # index of current draft token sample
                        sample = input_ids[:, draft_idx][0] # token id of sample
                        input_ids[:, draft_idx] = input_tokens[:, draft_idx] # copy original ShareGPT sample back
                        if agree:
                            cur_target_logit = target_logits[:, draft_idx - 1, :] # probability that target would've predicted sample
                            cur_target_prob = temperature_softmax(cur_target_logit, temp=temp)
                            if temp != 0: 
                                if draft_probs[j] > cur_target_prob[:, sample]:
                                    agree = False
                            else:
                                greedy_target_sample = torch.multinomial(cur_target_prob, 1)[:, 0]
                                if greedy_target_sample != sample:
                                    agree = False
                            num_tok_generated += 1
                
                    agreement_stats.append(num_tok_generated)
        
        avg_agreement = statistics.mean(agreement_stats)
        min_agreement = min(agreement_stats)
        max_agreement = max(agreement_stats)
        median_agreement = statistics.median(agreement_stats)
        std_agreement = statistics.stdev(agreement_stats) if len(agreement_stats) > 1 else 0.0
        percentiles = np.percentile(agreement_stats, [25, 50, 75])
        print(f"Average Tokens Generated: {avg_agreement:.5f}")
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
with open(f"agreement_stats_by_layer_{temps[0]}.json", "w") as outfile:
    json.dump(agreement_stats_all, outfile, indent=4)

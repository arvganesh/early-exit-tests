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
    "--kl_temperature",
    type=float,
    default=2.0,
    help="Temperature value for KL divergence."
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=24,
    help="Batch size for training."
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
    "--wandb",
    action="store_true",
    help="Use to enable logging to wandb."
)
parser.add_argument(
    "--lm_head_random_init",
    action="store_true",
    help="Randomly initializes the LM head to train."
)
parser.add_argument(
    "--report_perplexity",
    action="store_true",
    help="Report per token perplexity on the datasets."
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
args = parser.parse_args()

torch.manual_seed(args.seed)
numpy.random.seed(args.seed)
random.seed(args.seed)
#torch.use_deterministic_algorithms(True)

torch.set_float32_matmul_precision("high")
model = TruncatedLlama(args.model_path, 
                       early_exit_idx=args.target_layer,
                       lm_head_random_init=args.lm_head_random_init, 
                       use_flash_attn=False)
model.new_lm_head.load_state_dict(torch.load(args.weights_to_load, map_location=args.device))
model.eval()
# model = torch.compile(model) if args.device == "cuda" else model
model.to(args.device)

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.pad_token = tokenizer.eos_token

llama_model = AutoModelForCausalLM.from_pretrained(args.model_path)
llama_model.to(args.device)
llama_model.eval()

total = sum(p.numel() for p in llama_model.parameters())
print(f"total # in base llama: {total}") 

def get_untuned_perplexity(model, input_ids, attention_mask, labels, target_layer):
    # Get outputs from exit layer.
    with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
        output = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = output.hidden_states[target_layer + 1]

        # If we aren't early exiting, RMSNorm is already applied to the final transformer layer's hidden states.
        if target_layer + 1 == len(output.hidden_states) - 1:
            early_exit_logits = model.lm_head(hidden_states)
        else:
            early_exit_logits = model.lm_head(model.model.norm(hidden_states))
         
    # Compute perplexity (averaged across the entire batch).
    perplexity = F.cross_entropy(early_exit_logits.view(-1, early_exit_logits.size(-1)), labels.view(-1)).item()
    return perplexity

# evaluate average perplexity over test dataset
# Get data tensors, move to device.
if args.report_perplexity:
    with torch.no_grad():
        train, test, val = get_sharegpt_dataloaders(args.batch_size, tokenizer, args.max_length, generate_labels = True)
        #train, test, val = get_toy_dataloaders(args.batch_size, tokenizer, args.max_length, generate_labels = True)
        early_exit_ft_perp = 0
        early_exit_base_perp = 0
        baseline_llama_perp = 0
        total_tokens = 0
        print(f"Num validation examples: {len(val)}")
        for idx, batch in enumerate(val):
            input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
            input_ids, attention_mask, labels = input_ids.to(args.device), attention_mask.to(args.device), labels.to(args.device)
            with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels, loss_type="cross_entropy", keep_og_logits=True)

            batch_tokens = attention_mask.sum().item()

             # Early exit with unfinetuned llama head.
            early_exit_base_perp += get_untuned_perplexity(llama_model, input_ids, attention_mask, labels, args.target_layer) * batch_tokens

            # Early exit with finetuned head.  
            loss, logits = outputs["loss"], outputs["logits"]
            early_exit_ft_perp += loss.item() * batch_tokens
            
            # Baseline Llama (unmodified)
            og_logits = outputs["og_lm_logits"]
            og_loss = F.cross_entropy(og_logits.view(-1, og_logits.size(-1)), labels.view(-1)).item()
            baseline_llama_perp += og_loss * batch_tokens
            
            total_tokens += batch_tokens
            if idx % int(len(val) / 10) == 0:
                print(f"{(idx + 1) / len(val) * 100:.2f}% of the way there!")

    print(f"baseline llama: {baseline_llama_perp / total_tokens}, finetuned early-exit: {early_exit_ft_perp / total_tokens}, early_exit no finetuning: {early_exit_base_perp / total_tokens}")

if args.sample_from_models:
    prompt = "who is obama?"
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


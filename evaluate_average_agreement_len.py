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
numpy.random.seed(args.seed)
random.seed(args.seed)
#torch.use_deterministic_algorithms(True)

torch.set_float32_matmul_precision("high")
model = TruncatedLlama(args.model_path, 
                       early_exit_idx=args.target_layer,
                       lm_head_random_init=False,
                       use_flash_attn=args.flash_attn)
model.new_lm_head.load_state_dict(torch.load(args.weights_to_load, map_location=args.device))
model.eval()
# model = torch.compile(model) if args.device == "cuda" else model
model.to(args.device)

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

# evaluate average perplexity over test dataset
# Get data tensors, move to device.
with torch.no_grad():
    if args.toy_dataset:
        train, test, val = get_toy_dataloaders(1, tokenizer, args.max_length, generate_labels = True, nice_shape = False)
    else:
        train, test, val = get_sharegpt_dataloaders(1, tokenizer, args.max_length, generate_labels = True, nice_shape = False)
    total_agreement_tok = 0
    print(f"Num examples: {len(val)}")
    for idx, batch in enumerate(val):
        input_ids = batch["input_ids"]
        input_ids = input_ids.to(args.device)
        agreement_length = 0
        
        should_print = False
        if should_print:
            print(f"{(idx + 1) / len(val) * 100:.2f}% of the way there!")
            print(f"Initial prompt: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")

        while True:
            with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
                outputs = model(input_ids, keep_og_logits=True)

            # Early exit with finetuned head.  
            logits = outputs["logits"][:, -1, :] 
            ee_probs = temperature_softmax(logits, temp=args.softmax_temperature)
            ee_sample = torch.multinomial(ee_probs, 1) 
            
            # Baseline Llama (unmodified)
            og_logits = outputs["og_lm_logits"][:, -1, :]
            og_probs = temperature_softmax(og_logits, temp=args.softmax_temperature) 
            og_sample = torch.multinomial(og_probs, 1)

            if not torch.equal(ee_sample, og_sample):
                if should_print:
                    print(f"ee_tok: {ee_sample.item()} og_tok: {og_sample.item()}")
                    print(f"P({ee_sample.item()} in EE_MODEL) = {ee_probs[0, ee_sample.item()]}")
                    print(f"P({ee_sample.item()} in OG_MODEL) = {og_probs[0, ee_sample.item()]}")
                    print(f"P({og_sample.item()} in EE_MODEL) = {ee_probs[0, og_sample.item()]}")
                    print(f"P({og_sample.item()} in OG_MODEL) = {og_probs[0, og_sample.item()]}")
                break
           
            agreement_length += 1
            input_ids = torch.cat((input_ids, ee_sample), dim=1)
        
        if should_print:
            print(agreement_length)
        total_agreement_tok += agreement_length 

print(f"Average agreement length: {total_agreement_tok / len(val):.5f}")

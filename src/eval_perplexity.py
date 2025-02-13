import torch
import torch.nn.functional as F
import argparse
from truncated_llama import TruncatedLlama
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup
from share_gpt_dataset import get_sharegpt_dataloaders

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
args = parser.parse_args()

torch.set_float32_matmul_precision("high")
model = TruncatedLlama(args.model_path, 
                       early_exit_idx=args.target_layer,
                       lm_head_random_init=args.lm_head_random_init, 
                       use_flash_attn=False)
model.new_lm_head.load_state_dict(torch.load(args.weights_to_load))
model.eval()
# model = torch.compile(model) if args.device == "cuda" else model
model.to(args.device)

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.pad_token = tokenizer.eos_token

# train, test, val = get_sharegpt_dataloaders(args.batch_size, tokenizer, args.max_length, generate_labels = True)

# evaluate average perplexity over test dataset
# Get data tensors, move to device.
# with torch.no_grad():
#     early_exit_perplexity = 0
#     expected_perplexity = 0
#     total_tokens = 0
#     for idx, batch in enumerate(val):
#         input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
#         input_ids, attention_mask, labels = input_ids.to(args.device), attention_mask.to(args.device), labels.to(args.device)
#         with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
#             outputs = model(input_ids, attention_mask=attention_mask, labels=labels, loss_type="cross_entropy", keep_og_logits=True)

#         loss, logits = outputs["loss"], outputs["logits"]
#         batch_tokens = attention_mask.sum().item()
#         early_exit_perplexity += loss.item() * batch_tokens
        
#         og_logits = outputs["og_lm_logits"]
#         og_loss = F.cross_entropy(og_logits.view(-1, og_logits.size(-1)), labels.view(-1)).item()
#         expected_perplexity += og_loss * batch_tokens
#         total_tokens += batch_tokens
#         if idx % 1000 == 0:
#             print("almost there")
# print(f"exp: {expected_perplexity / total_tokens}, actual: {early_exit_perplexity / total_tokens}")

prompt = "Hello! I'm a language model."
inputs = tokenizer(prompt, return_tensors="pt")
og_input_ids = inputs["input_ids"]
og_input_ids = og_input_ids.to(args.device)

inputs = tokenizer(prompt, return_tensors="pt")
actual_input_ids = inputs["input_ids"]
actual_input_ids = actual_input_ids.to(args.device)

llama_model = AutoModelForCausalLM.from_pretrained(args.model_path)
llama_model.to(args.device)
llama_model.eval()

my_outputs = model.generate(actual_input_ids, 20, tokenizer.eos_token_id)
llama_outputs = llama_model.generate(input_ids=og_input_ids, max_new_tokens=20)

my_text = tokenizer.decode(my_outputs[0], skip_special_tokens=True)
llama_text = tokenizer.decode(llama_outputs[0], skip_special_tokens=True)

print(my_text)
print(" ------ ")
print(llama_text)

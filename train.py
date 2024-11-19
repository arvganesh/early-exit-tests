from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=True)

# Set padding token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples['prompt'], examples['response'], truncation=True)

# Tokenize your dataset
raw_dataset = [
    {"prompt": "User: How are you?", "response": "Assistant: I'm good, thank you!"},
    {"prompt": "User: What's the weather today?", "response": "Assistant: It's sunny and warm."}
]
tokenized_dataset = map(tokenize_function, raw_dataset)
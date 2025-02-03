import torch
import os
from transformers import LlamaTokenizer

from truncated_llama import TruncatedLlama

device = "cuda" if torch.cuda.is_available() else "cpu"

def eval(
    model_path: str = "meta-llama/Llama-2-7b-hf",
    target_layer: int = 16,
):
    # Setup Tokenizer, load model, move to device.
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = TruncatedLlama(
        model_path,
        target_layer
    )
    weights = torch.load("./truncated_llama_on_slimPJ6B/llama-trunc-350step")
    model.model.lm_head.load_state_dict(weights["trained_params"])
    model.eval()
    model.to(device)

    print(f"Loaded model {model_path} to {device}")

    # Manual Evaluation
    input_str = "Hello"
    input_ids = tokenizer.encode(input_str, return_tensors="pt")
    input_ids = input_ids.to(device)
    print(input_ids)
    for i in range(5):
        outputs = model.generate(input_ids, max_length=20)
        print(tokenizer.batch_decode(outputs, skip_special_tokens=False))

if __name__ == "__main__":
    eval()
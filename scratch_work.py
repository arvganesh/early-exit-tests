import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from truncated_llama import TruncatedLlama

device = "cuda" if torch.cuda.is_available() else "cpu"

def eval(
    model_path: str = "meta-llama/Llama-3.2-1B",
    target_layer: int = 16,
):
    # Setup Tokenizer, load model, move to device.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    # model = TruncatedLlama(
    #     model_path,
    #     target_layer
    # )
    # weights = torch.load("./truncated_llama_on_slimPJ6B/llama-trunc-350step")
    # model.model.lm_head.load_state_dict(weights["trained_params"])
    model.eval()
    model.to(device)

    print(f"Loaded model {model_path} to {device}")
    print(tokenizer.bos_token)

    # Manual Evaluation
    input_str = "Hi! I'm a language model."
    input_ids = tokenizer.__call__(input_str, return_tensors="pt")
    import pdb; pdb.set_trace()
    input_ids = input_ids.to(device)
    for i in range(1):
        outputs = model.generate(input_ids["input_ids"], attention_mask=input_ids["attention_mask"])
        print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

if __name__ == "__main__":
    eval()
from tuned_llama import LlamaWithTunedHead
from safetensors.torch import load_model
from transformers import LlamaTokenizer
import torch
import os
from peft import get_peft_model, LoraConfig, PeftModel
device = "cuda" if torch.cuda.is_available() else "cpu"

def eval(
    model_path: str = "meta-llama/Llama-2-7b-hf",
    target_layer: int = 16,
    loss_type: str = "perplexity",
    kl_temperature: float = 2.0,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    num_epochs: int = 3,
    max_length: int = 512,
    gradient_accumulation_steps: int = 2,
    output_dir: str = "llama_tuned_head_output/checkpoint-1000",
    use_lora: bool = True
):
    # Setup Tokenizer, load model, move to device.
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = LlamaWithTunedHead(
        model_path,
        target_layer,
        loss_type=loss_type,
        kl_temperature=kl_temperature,
        dtype=torch.bfloat16,
        device=device
    )

    if use_lora:
        model = PeftModel.from_pretrained(model, output_dir)

    model.to(device)
    model.eval()

    print(f"Loaded model {model_path} to {device}")
    print(f"Number of trainable parameters: {model.num_trainable_params()}")

    # Manual Evaluation
    input_str = "Hello, how are you?"
    input_ids = tokenizer(input_str, return_tensors="pt").input_ids.to(device)
    for i in range(10):
        outputs = model.generate(input_ids, max_length=10)
        print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

if __name__ == "__main__":
    eval()
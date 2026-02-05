import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from truncated_llama import TruncatedLlama


def compare_hidden_states(model_path: str, layer_idx: int, device: str, prompt: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    base_model.to(device)
    base_model.eval()

    truncated = TruncatedLlama(
        model_path,
        early_exit_idx=layer_idx,
        use_flash_attn=False,
        ft_last_transformer=False,
        ft_head=False,
        lm_head_random_init=False,
    )
    truncated.to(device)
    truncated.eval()

    with torch.no_grad():
        base_out = base_model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        trunc_out = truncated.truncated_model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )

    diffs = []
    print(f"Truncated Model HS len {len(trunc_out.hidden_states)}")
    print(f"Base Model HS len {len(base_out.hidden_states)}")
    for idx, trunc_state in enumerate(trunc_out.hidden_states):
        ref_state = base_out.hidden_states[idx]
        max_diff = (ref_state - trunc_state).abs().max().item()
        diffs.append((idx, max_diff))

    return diffs


def main():
    parser = argparse.ArgumentParser(description="Compare hidden states between HF and truncated models.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--target_layer", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt", type=str, default="Hello, world!")
    args = parser.parse_args()

    torch.manual_seed(0)
    diffs = compare_hidden_states(args.model_path, args.target_layer, args.device, args.prompt)
    for idx, diff in diffs:
        print(f"Layer {idx}: max abs diff {diff:.3e}")


if __name__ == "__main__":
    main()

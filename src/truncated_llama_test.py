import torch
import numpy as np
import random
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from pprint import pprint
from truncated_llama import TruncatedLlama
from data_utils import custom_collate_fn, get_toy_dataloaders

parser = argparse.ArgumentParser(prog="Early Exiting Llama Testing")
parser.add_argument("-d", "--device", type=str)

# Make sure results are deterministic.
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def test_same_output_with_last_layer_exit():
    model_path = "meta-llama/llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    truth_model = AutoModelForCausalLM.from_pretrained(model_path)
    num_original_layers = len(truth_model.model.layers) - 1
    print(f"Num OG Layers: {num_original_layers}")
    test_model = TruncatedLlama(model_path,
                                early_exit_idx=num_original_layers,
                                lm_head_random_init=False,
                                use_flash_attn=False)
    prompt = "Hello!"
    inputs = tokenizer(prompt, return_tensors="pt")
    fake_inputs = tokenizer(prompt, return_tensors="pt")
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        truth_outputs = truth_model(inputs["input_ids"])
        test_outputs = test_model(fake_inputs["input_ids"], loss_type="kl_divergence")

    truth_logits, actual_logits = truth_outputs["logits"], test_outputs["logits"]
    test_loss = test_outputs["loss"]

    assert truth_logits.shape == actual_logits.shape
    assert (truth_logits == actual_logits).all()
    assert test_loss == 0

def test_hidden_states_match_on_early_exit():
    model_path = "meta-llama/llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    truth_model = AutoModelForCausalLM.from_pretrained(model_path)
    early_exit_layer = (len(truth_model.model.layers) - 1) // 2 # exit at the halfway point
    test_model = TruncatedLlama(model_path,
                                early_exit_idx=early_exit_layer,
                                lm_head_random_init=False,
                                use_flash_attn=False)
    prompt = "Hello!"
    inputs = tokenizer(prompt, return_tensors="pt")
    fake_inputs = tokenizer(prompt, return_tensors="pt")

    # Still need to apply RMS norm to intermediate outputs.
    truth_outputs = truth_model(inputs["input_ids"], output_hidden_states=True)
    truth_hidden_states = truth_outputs.hidden_states[early_exit_layer + 1]

    expected_logits = truth_model.lm_head(truth_model.model.norm(truth_hidden_states))
    actual_logits = test_model(fake_inputs["input_ids"])["logits"]

    assert expected_logits.shape == actual_logits.shape
    assert (expected_logits == actual_logits).all()


def test_collate():
    test_name = "test_collate()"
    model_path = "meta-llama/llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Expect that batch to collate is padded to max length item.
    batch = [
        {
        'input_ids': torch.Tensor([128000, 284, 12, 53, 88]), 
        'attention_mask': torch.Tensor([1, 1, 1, 1, 1]),
        },
        {
        'input_ids': torch.Tensor([128000, 123]), 
        'attention_mask': torch.Tensor([1, 1]),
        }
    ]
    
    collated_batch = custom_collate_fn(batch, tokenizer)
    pprint(collated_batch)

    # Assert that padding happened
    expected_shape = (len(batch), 8) # Round up to nearest power of two.
    for key in collated_batch:
        assert collated_batch[key].shape == expected_shape

    # Assert that padding tokens are correct.
    assert (collated_batch["labels"][0, :] == torch.Tensor([284, 12, 53, 88, -100])).all()
    assert (collated_batch["labels"][1, :] == torch.Tensor([123, -100, -100, -100, -100])).all()

def test_toy_dataloader():
    model_path = "meta-llama/llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    batch_size = 2
    max_length = 64
    train, test, val = get_toy_dataloaders(batch_size, tokenizer, max_length, generate_labels = True)
    train_batch = next(iter(train))
    input_ids = train_batch["input_ids"]
    assert input_ids.size(0) == batch_size
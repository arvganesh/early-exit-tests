import torch
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from pprint import pprint
from truncated_llama import TruncatedLlama
from data_utils import custom_collate_fn

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
    truth_logits = truth_model(inputs["input_ids"]).logits
    actual_logits = test_model(fake_inputs["input_ids"]).logits

    assert truth_logits.shape == actual_logits.shape
    assert (truth_logits == actual_logits).all()

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

    truth_outputs = truth_model(inputs["input_ids"], output_hidden_states=True)
    truth_hidden_states = truth_outputs.hidden_states[early_exit_layer + 1]

    expected_logits = truth_model.lm_head(truth_hidden_states)
    actual_logits = test_model(fake_inputs["input_ids"]).logits

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
    expected_shape = (len(batch), 5)
    for key in collated_batch:
        assert collated_batch[key].shape == expected_shape

    # Assert that padding tokens are correct.
    assert (collated_batch["labels"][0, :] == torch.Tensor([284, 12, 53, 88, -100])).all()
    assert (collated_batch["labels"][1, :] == torch.Tensor([123, -100, -100, -100, -100])).all()

    print(f"Passed {test_name}")
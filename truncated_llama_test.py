import torch
import numpy as np
import random
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from pprint import pprint
from truncated_llama import TruncatedLlama, masked_kl_loss
from data_utils import custom_collate_fn, get_toy_dataloaders

parser = argparse.ArgumentParser(prog="Early Exiting Llama Testing")
parser.add_argument("-d", "--device", type=str)

# Make sure results are deterministic.
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)

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
    assert torch.allclose(truth_logits, actual_logits, atol=1e-5, rtol=1e-5)
    assert test_loss is not None
    assert float(test_loss) < 1e-5

def test_output_same_for_ft_last_transformer():
    model_path = "meta-llama/llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    test_model = TruncatedLlama(model_path,
                                early_exit_idx=15,
                                lm_head_random_init=False,
                                use_flash_attn=False,
                                ft_last_transformer=False,
                                ft_head=False)

    # Set Q_proj to all zeroes to mess up the output of the early exit layer.
    # Assert that the model's calculation of llama's original logits remain correct.
    with torch.no_grad():
        test_model.early_exit_layer.self_attn.q_proj.weight.zero_()

    truth_model = AutoModelForCausalLM.from_pretrained(model_path)
                                
    prompt = "Hello!"
    inputs = tokenizer(prompt, return_tensors="pt")
    fake_inputs = tokenizer(prompt, return_tensors="pt")

    print(inputs["input_ids"])
    print(fake_inputs["input_ids"])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        truth_outputs = truth_model(inputs["input_ids"])
        test_outputs = test_model(fake_inputs["input_ids"], loss_type="kl_divergence", keep_og_logits=True)

    truth_logits, og_logits = truth_outputs["logits"], test_outputs["og_lm_logits"]
    assert torch.allclose(truth_logits, og_logits, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(truth_logits, test_outputs["logits"], atol=1e-5, rtol=1e-5)

def test_masked_kl_loss():
    """
    B: Batch size
    T: Sequence Length
    C: Vocab Size
    """
    B, T, C = 2, 2, 4

    student_logits = torch.rand(B, T, C, dtype=torch.float32) 
    teacher_logits = torch.rand(B, T, C, dtype=torch.float32) 
    truth_KL = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    
    # No masking should match regular batchmean KL div loss. 
    all_ones = torch.ones(B, T)
    my_loss = masked_kl_loss(student_logits, teacher_logits, all_ones)
    actual_loss = truth_KL(student_logits, teacher_logits)
    assert torch.isclose(my_loss, actual_loss)
    
    # With masking, we should remove certain entries.
    all_zeros = torch.zeros(B, T) 
    assert masked_kl_loss(student_logits, teacher_logits, all_zeros) == 0

    # Add one entry back via mask, and it should be non-zero. 
    all_zeros[0, 1] = 1
    mask = all_zeros
    loss = masked_kl_loss(student_logits, teacher_logits, mask)
    actual_loss = truth_KL(student_logits[0, 1], teacher_logits[0, 1])
    assert loss == actual_loss * 2

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
        'loss_mask': torch.Tensor([0, 0, 1, 1, 0]),
        },
        {
        'input_ids': torch.Tensor([128000, 123]), 
        'attention_mask': torch.Tensor([1, 1]),
        'loss_mask': torch.Tensor([0, 1]),
        }
    ]
    
    collated_batch = custom_collate_fn(batch, tokenizer, nice_shape=False)
    pprint(collated_batch)

    # Assert that padding happened
    expected_shape = (len(batch), 5) # Round up to nearest power of two.
    for key in collated_batch:
        assert collated_batch[key].shape == expected_shape

    # Assert that padding tokens are correct.
    assert (collated_batch["labels"][0, :] == torch.Tensor([284, 12, 53, 88, -100])).all()
    assert (collated_batch["labels"][1, :] == torch.Tensor([123, -100, -100, -100, -100])).all()
    assert torch.equal(collated_batch["loss_mask"][1, :], torch.Tensor([1, 0, 0, 0, 0]))

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

if __name__ == "__main__":
    test_output_same_for_ft_last_transformer()

from MutantLlama import MutantLlama, EarlyExitConfig, LLAMA32_CONFIG_1B, robust_copy_weights
from transformers import AutoModelForCausalLM
import torch

torch.set_float32_matmul_precision('high')
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

# pytest compatible tests
reference_model = "meta-llama/Llama-3.2-1B"
device = "mps"

# MutantLlama without EE Config mirrors the original llama model
def test_mutant_llama():
    # Load the model
    model = MutantLlama(LLAMA32_CONFIG_1B).to(device)

    # Load the reference model
    hf_model = AutoModelForCausalLM.from_pretrained(reference_model).to(device)

    # Copy weights from the reference model to the mutant model
    model.load_from_hf(hf_model)

    # Test the model
    test_input = torch.randint(0, 128_256, (1, 10)).to(device)
    test_output = model(test_input)
    true_output = hf_model(test_input)

    # Check if the output is the same
    assert torch.allclose(test_output, true_output)

def test_mutant_llama_with_ee_config():
    # Load the model
    ee_cfg = EarlyExitConfig(cfg=LLAMA32_CONFIG_1B, layer_idx=10, train_transformer=False)
    model = MutantLlama(LLAMA32_CONFIG_1B, ee_cfg=ee_cfg).to(device)

    # Load the reference model
    hf_model = AutoModelForCausalLM.from_pretrained(reference_model).to(device)

    # Copy weights from the reference model to the mutant model
    model.load_from_hf(hf_model)

    # Test the model
    test_input = torch.randint(0, 128_256, (1, 10)).to(device)
    test_output = model(test_input)
    true_output = hf_model(test_input)

    # Check if the output is the same
    assert torch.allclose(test_output["llama_logits"], true_output["logits"])
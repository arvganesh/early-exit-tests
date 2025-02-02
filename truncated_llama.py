import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, Trainer, TrainingArguments
from transformers.modeling_outputs import MaskedLMOutput

class TruncatedLlama(nn.Module):
    def __init__(self, model_path: str, num_transformer_layers: int):
        super().__init__()
        # self.model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        self.model = LlamaForCausalLM.from_pretrained(model_path)
        self.model.model.layers = self.model.model.layers[:num_transformer_layers]
        # self.model.lm_head = nn.Linear(self.model.lm_head.in_features, self.model.lm_head.out_features, bias=False, dtype=torch.bfloat16)
        self.model.lm_head = nn.Linear(self.model.lm_head.in_features, self.model.lm_head.out_features, bias=False)

        # Freeze all parameters except the new LM head
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.lm_head.parameters():
            param.requires_grad = True

        # Print the number of parameters in the model
        print(f"Number of trainable parameters in the model: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        print(f"Number of parameters in the model: {sum(p.numel() for p in self.model.parameters())}")


    def forward(self, input_ids: torch.Tensor, attention_mask=None, labels=None):
        # Get logits from model
        outputs = self.model(input_ids, attention_mask=attention_mask)

        # Remove first token from labels, last token from logits
        labels = labels[:, 1:].contiguous()
        logits = outputs.logits[:, :-1, :].contiguous()

        # Get loss
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return MaskedLMOutput(
                loss=loss,
                logits=logits
            )
    
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and p.requires_grad]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 or not p.requires_grad]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        use_fused = device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

if __name__ == "__main__":
    model = TruncatedLlama("meta-llama/Llama-2-7b-hf", 16)
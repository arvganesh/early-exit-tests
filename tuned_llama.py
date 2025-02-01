from typing import Literal, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM

class LlamaWithTunedHead(nn.Module):
    def __init__(
        self, 
        base_model_path: str, 
        target_layer: int,
        loss_type: Literal["perplexity", "kl_divergence", "combined"] = "perplexity",
        kl_temperature: float = 2.0,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda"
    ):
        super().__init__()
        self.model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            torch_dtype=dtype
        )
        self.target_layer = target_layer
        self.loss_type = loss_type
        self.kl_temperature = kl_temperature
        self.device = device
        
        # Create a copy of the LM head for the target layer
        self.target_lm_head = nn.Linear(
            self.model.lm_head.in_features,
            self.model.lm_head.out_features,
            bias=False,
            dtype=dtype
        )

        # Initialize with the same weights as the original LM head
        self.target_lm_head.weight.data = self.model.lm_head.weight.data.clone()
        
        # Freeze all parameters except the target LM head and target layer
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        
        # Ensure the target LM head is trainable
        for param in self.target_lm_head.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        # Get intermediate hidden states and final logits
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        
        loss = None
        metrics = {}
        
        # Get logits from target layer
        target_hidden = outputs.hidden_states[self.target_layer]
        target_logits = self.target_lm_head(target_hidden)
        
        # Prepare shifted sequences for loss computation
        shift_logits = target_logits[..., :-1, :].contiguous()
        full_logits = outputs.logits[..., :-1, :].contiguous()
            
        # Compute losses based on selected type
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            if self.loss_type in ["perplexity", "combined"]:
                loss_fct = nn.CrossEntropyLoss()
                loss_a = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                shift_labels.view(-1))
                metrics["perplexity_loss"] = loss_a.item()
                
                if self.loss_type == "perplexity":
                    loss = loss_a
                
            if self.loss_type in ["kl_divergence", "combined"]:
                kl_loss = nn.KLDivLoss(reduction="batchmean")
                loss_b = kl_loss(
                    F.log_softmax(shift_logits / self.kl_temperature, dim=-1),
                    F.softmax(full_logits / self.kl_temperature, dim=-1)
                )
                metrics["kl_loss"] = loss_b.item()
                
                if self.loss_type == "kl_divergence":
                    loss = loss_b
            
            if self.loss_type == "combined":
                loss = loss_a + loss_b
        
        return {
            "loss": loss,
            "target_logits": target_logits,
            "full_logits": outputs.logits,
            "metrics": metrics
        }

    def generate(self, input_ids: torch.Tensor, max_length: int = 512) -> torch.Tensor:
        output = []
        for i in range(max_length):
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=None,
                labels=None
            )
            logits = outputs["target_logits"]
            logits = logits.view(-1, logits.size(-1))
            logits = logits[-1].softmax(dim=-1)
            sampled_token = torch.multinomial(logits, num_samples=1)
            input_ids = torch.cat([input_ids, sampled_token.view(1, 1)], dim=-1)
            output.append(sampled_token)
        return input_ids

    # Get # of trainable parameters
    def num_trainable_params(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable_params

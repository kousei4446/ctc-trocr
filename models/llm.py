from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn


class LLM(nn.Module):
    """
    Wrapper for HuggingFace causal language models (LLaMA).
    Inputs:
        - inputs_embeds: (B, T, D)
        - labels: (B, T)
        - attention_mask: (B, T) optional
    Outputs:
        - logits: (B, T, V)
    """

    def __init__(self, model_name: str, freeze: bool = True):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Llama 等も含めて hidden_size に統一
        self.input_dim = self.model.config.hidden_size
        self.config = self.model.config
        if freeze:
            self.model.requires_grad_(False)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask.to(self.model.device) if attention_mask is not None else None,
            use_cache=False,
            return_dict=True,
        )
        return outputs

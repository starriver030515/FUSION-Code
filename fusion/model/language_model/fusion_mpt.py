from typing import Optional, Tuple

import torch

from transformers import AutoConfig, AutoModelForCausalLM, \
                         MptConfig, MptForCausalLM, MptModel
from fusion.model.fusion_arch import FusionMetaModel, FusionMetaForCausalLM


class FusionMptConfig(MptConfig):
    model_type = "fusion_mpt"


class FusionMptModel(FusionMetaModel, MptModel):
    config_class = FusionMptConfig

    def __init__(self, config: MptConfig):
        config.hidden_size = config.d_model
        super(FusionMptModel, self).__init__(config)
    
    def embed_tokens(self, x):
        return self.wte(x)


class FusionMptForCausalLM(MptForCausalLM, FusionMetaForCausalLM):
    config_class = FusionMptConfig
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super(MptForCausalLM, self).__init__(config)

        self.transformer = FusionMptModel(config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.transformer

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, FusionMptModel):
            module.gradient_checkpointing = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images=None):

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        
        return super().forward(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        _inputs['images'] = images
        return _inputs


AutoConfig.register("fusion_mpt", FusionMptConfig)
AutoModelForCausalLM.register(FusionMptConfig, FusionMptForCausalLM)

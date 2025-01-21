from transformers import (
    LlamaForCausalLM,
    AutoConfig,
)
from config import PrepareConfig, ParentModelConfig

def prepare_models():
    parent_llama = LlamaForCausalLM.from_pretrained(ParentModelConfig.pretrained_model_name)

    child_config = AutoConfig.from_pretrained(ParentModelConfig.pretrained_model_name, num_hidden_layers=PrepareConfig.num_hidden_layers)
    layers = PrepareConfig.layers
    child_llama = LlamaForCausalLM(child_config)

    for i in range(PrepareConfig.num_hidden_layers):
        child_llama.model.layers[i].load_state_dict(parent_llama.model.layers[layers[i]].state_dict())
    
    return (child_llama, parent_llama)
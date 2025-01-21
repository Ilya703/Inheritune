from transformers import (
    LlamaForCausalLM,
    AutoConfig,
)
from config import prepare_config, parent_model_config

def prepare_models():
    parent_llama = LlamaForCausalLM.from_pretrained(parent_model_config.pretrained_model_name)

    child_config = AutoConfig.from_pretrained(parent_model_config.pretrained_model_name, num_hidden_layers=prepare_config.num_hidden_layers)
    layers = prepare_config.layers
    child_llama = LlamaForCausalLM(child_config)

    for i in range(prepare_config.num_hidden_layers):
        child_llama.model.layers[i].load_state_dict(parent_llama.model.layers[layers[i]].state_dict())
    
    return (child_llama, parent_llama)
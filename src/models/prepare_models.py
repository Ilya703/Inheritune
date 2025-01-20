from transformers import (
    LlamaForCausalLM,
    AutoConfig,
)

def prepare_models():
    parent_llama = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b_v2")

    child_config = AutoConfig.from_pretrained("openlm-research/open_llama_3b_v2", num_hidden_layers=6)
    layers = [0, 3, 10, 15, 22, 25]
    child_llama = LlamaForCausalLM(child_config)

    for i in range(6):
        child_llama.model.layers[i].load_state_dict(parent_llama.model.layers[layers[i]].state_dict())
    
    return (child_llama, parent_llama)
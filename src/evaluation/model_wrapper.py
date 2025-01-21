'''
    Обертка для модели, необходимая для оценки эффективности 
    с помощью библиотеки lm-evaluation-harness
'''
from lm_eval import BaseLM
import torch
from typing import List
import gc
from config import wrapper_config

class ModelWrapper(BaseLM):
    def __init__(
        self,
        model,
        batch_size,
        tokenizer,
        device
    ):
        super().__init__()
        self.config = model.config
        self.model = model
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(device)
        self.tokenizer = tokenizer
        self.batch_size_per_gpu = batch_size
        self.device_ = device

    @torch.inference_mode()
    def _model_call(self, inps):
        outputs = self.model(inps)
        if hasattr(outputs, 'logits'):
            return outputs.logits
        elif hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state
        else:
            raise ValueError("Model output does not contain 'logits' or 'last_hidden_state'")

    @torch.inference_mode()
    def _model_generate(self, context, max_length, eos_token_id) -> torch.Tensor:
        # this only supports batch size 1
        assert context.shape[0] == 1
        out = generate(self.model, context[0], max_length, eos_id=eos_token_id)
        for block in self.model.transformer.h:
            block.attn.kv_cache.reset_parameters()
        return out.unsqueeze(0)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu*torch.cuda.device_count()

    @property
    def device(self):
        return self.device_

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_id

    @property
    def max_gen_toks(self):
        return wrapper_config.max_toks

    @property
    def max_length(self):
        return self.config.max_position_embeddings

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: List[int]) -> str:
        t = torch.tensor(tokens)
        return self.tokenizer.decode(t)
    
    def clear_gpu_memory(self):
        self.model.module.cpu()
        del self.model  
        gc.collect()
        torch.cuda.empty_cache()
        for device in range(torch.cuda.device_count()):
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
        gc.collect()
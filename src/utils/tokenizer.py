from transformers import LlamaTokenizer
from config import TokenizerConfig, ParentModelConfig

tokenizer=LlamaTokenizer.from_pretrained(ParentModelConfig.pretrained_model_name)
tokenizer.pad_token_id = 0

'''
    Функция для токенизации датасета
'''
def tokenize_function(examples):
    texts = [text for text in examples['text'] if text is not None]
    return tokenizer(
        texts,
        return_special_tokens_mask=True,
        truncation=True,
        max_length=TokenizerConfig.max_length,
        padding='max_length'
    )
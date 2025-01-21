from transformers import LlamaTokenizer
from config import tokenizer_config, parent_model_config

tokenizer=LlamaTokenizer.from_pretrained(parent_model_config.pretrained_model_name)
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
        max_length=tokenizer_config.max_length,
        padding='max_length'
    )
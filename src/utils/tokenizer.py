
from transformers import LlamaTokenizer

tokenizer=LlamaTokenizer.from_pretrained(
        "openlm-research/open_llama_3b_v2",
    )
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
        max_length=2048,
        padding='max_length'
    )
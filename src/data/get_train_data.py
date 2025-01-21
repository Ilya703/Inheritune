from datasets import load_dataset, concatenate_datasets
from src.utils.dataset import filter_dataset
from src.utils.tokenizer import tokenize_function
from src.data.custom_dataset import CustomDataset

def get_train_data():
    wikipedia_dataset = load_dataset('wikipedia', '20220301.en', split='train[:1%]')
    bookcorpus_dataset = load_dataset('bookcorpus', split='train[:1%]')

    wikipedia_dataset_filtered = filter_dataset(wikipedia_dataset, 10)
    bookcorpus_dataset_filtered = filter_dataset(bookcorpus_dataset, 10)

    tokenized_wikipedia_dataset = wikipedia_dataset_filtered.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    tokenized_bookcorpus_dataset = bookcorpus_dataset_filtered.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    combined_dataset = concatenate_datasets([tokenized_wikipedia_dataset, tokenized_bookcorpus_dataset])

    train_dataset = CustomDataset(combined_dataset)

    return train_dataset


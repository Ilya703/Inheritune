from datasets import load_dataset, concatenate_datasets
from src.utils.dataset import filter_dataset
from src.utils.tokenizer import tokenize_function
from src.data.custom_dataset import CustomDataset
from config import dataset_config

def get_train_data():
    wikipedia_dataset = load_dataset(
        dataset_config.wikipedia_dataset_name,
        dataset_config.wikipedia_dataset_version,
        split=dataset_config.dataset_split
    )
    bookcorpus_dataset = load_dataset(
        dataset_config.bookcorpus_dataset_name,
        split=dataset_config.dataset_split
    )

    wikipedia_dataset_filtered = filter_dataset(wikipedia_dataset, dataset_config.filter_threshold)
    bookcorpus_dataset_filtered = filter_dataset(bookcorpus_dataset, dataset_config.filter_threshold)

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

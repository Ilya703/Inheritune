from datasets import load_dataset, concatenate_datasets
from src.utils.dataset import filter_dataset
from src.utils.tokenizer import tokenize_function
from src.data.custom_dataset import CustomDataset
from config import DatasetConfig

def get_train_data():
    wikipedia_dataset = load_dataset(
        DatasetConfig.wikipedia_dataset_name,
        DatasetConfig.wikipedia_dataset_version,
        split=DatasetConfig.dataset_split
    )
    bookcorpus_dataset = load_dataset(
        DatasetConfig.bookcorpus_dataset_name,
        split=DatasetConfig.dataset_split
    )

    wikipedia_dataset_filtered = filter_dataset(wikipedia_dataset, DatasetConfig.filter_threshold)
    bookcorpus_dataset_filtered = filter_dataset(bookcorpus_dataset, DatasetConfig.filter_threshold)

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

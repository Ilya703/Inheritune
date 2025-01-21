from transformers import (
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from src.utils.tokenizer import tokenizer
from src.data.get_train_data import get_train_data
from src.models.prepare_models import prepare_models
from config import trained_model_config, train_inheritune_config

BATCH_SIZE = train_inheritune_config.batch_size
NUM_EPOCHS = train_inheritune_config.num_epochs
PER_DEVICE_TRAIN_BATCH_SIZE = train_inheritune_config.per_device_train_batch_size
from src.utils.set_seed import set_seed

def train():
    set_seed(42)

    child_llama, _ = prepare_models()

    training_args = TrainingArguments(
        output_dir=train_inheritune_config.training_args_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        save_steps=train_inheritune_config.save_steps,
        save_total_limit=train_inheritune_config.save_total_limit,
        prediction_loss_only=True,
        fp16=True,
        gradient_accumulation_steps=train_inheritune_config.gradient_accumulation_steps,
        report_to=None,
        dataloader_num_workers=train_inheritune_config.dataloader_num_workers
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=child_llama,
        args=training_args,
        data_collator=data_collator,
        train_dataset=get_train_data(),
    )

    trainer.train()

    trainer.save_model(trained_model_config.model_path)
    tokenizer.save_pretrained(trained_model_config.model_path)

if __name__ == "__main__":
    train()
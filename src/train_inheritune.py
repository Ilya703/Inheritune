from transformers import (
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from src.utils.tokenizer import tokenizer
from src.data.get_train_data import get_train_data
from Inheritune.src.models.prepare_models import prepare_models

BATCH_SIZE = 8
NUM_EPOCHS = 6
PER_DEVICE_TRAIN_BATCH_SIZE = 8

def train():
    child_llama, _ = prepare_models()

    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=True,
        gradient_accumulation_steps=8,
        report_to=None,
        dataloader_num_workers=4
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

    trainer.save_model("./open_llama_3b_v2_first_last_step")
    tokenizer.save_pretrained("./open_llama_3b_v2_first_last_step")

if __name__ == "__main__":
    train()
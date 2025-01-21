from transformers import (
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from src.utils.tokenizer import tokenizer
from src.data.get_train_data import get_train_data
from Inheritune.src.models.prepare_models import prepare_models
from config import TrainedModelConfig, TrainInherituneConfig

BATCH_SIZE = TrainInherituneConfig.batch_size
NUM_EPOCHS = TrainInherituneConfig.num_epochs
PER_DEVICE_TRAIN_BATCH_SIZE = TrainInherituneConfig.per_device_train_batch_size

def train():
    child_llama, _ = prepare_models()

    training_args = TrainingArguments(
        output_dir=TrainInherituneConfig.training_args_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        save_steps=TrainInherituneConfig.save_steps,
        save_total_limit=TrainInherituneConfig.save_total_limit,
        prediction_loss_only=True,
        fp16=True,
        gradient_accumulation_steps=TrainInherituneConfig.gradient_accumulation_steps,
        report_to=None,
        dataloader_num_workers=TrainInherituneConfig.dataloader_num_workers
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

    trainer.save_model(TrainedModelConfig.model_path)
    tokenizer.save_pretrained(TrainedModelConfig.model_path)

if __name__ == "__main__":
    train()
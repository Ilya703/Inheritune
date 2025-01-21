from Inheritune.src.models.prepare_models import prepare_models
from src.utils.distillation_loss import distillation_loss
from src.data.get_train_data import get_train_data
import time
import torch
from torch.utils.data import DataLoader
from src.utils.tokenizer import tokenizer
from transformers import AdamW

BATCH_SIZE = 8
NUM_EPOCHS = 4
STEP_LOG_PERIOD = 50

def train():
    student_llama, teacher_llama = prepare_models()

    step = 0
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = get_train_data()

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = AdamW(student_llama.parameters(), lr=3e-4)

    for _ in range(NUM_EPOCHS):
        student_llama.train()
        for inputs in train_loader:
            inputs = inputs.to(device)
            with torch.no_grad():
                teacher_outputs = teacher_llama(inputs).logits
            student_outputs = student_llama(inputs).logits

            loss = distillation_loss(student_outputs, teacher_outputs, temperature=2.0)
            if (step % STEP_LOG_PERIOD == 0):
                end_time = time.time()
                execution_time = end_time - start_time
                print(f'step: {step} / {len(train_loader)*NUM_EPOCHS}, loss: {loss}, execution time: {execution_time}')
                start_time = time.time()
            step += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    student_llama.save_pretrained("./distillation_llama")
    tokenizer.save_pretrained("./open_llama_3b_v2_first_last_step")

if __name__ == "__main__":
    train()

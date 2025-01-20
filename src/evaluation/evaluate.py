from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from typing import List
from src.evaluation.model_wrapper import ModelWrapper
from lm_eval import evaluator, tasks
from src.models.prepare_models import prepare_models

BATCH_SIZE = 8

def evaluate():
    eval_tasks: List[str] = ['winogrande','boolq','piqa']

    num_fewshot = 0
    limit = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    child_model, parent_model = prepare_models()

    wrapped_model = ModelWrapper(
        model=parent_model,
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        device=device
    )

    results_parent = evaluator.evaluate(
        lm=wrapped_model,
        task_dict=tasks.get_task_dict(eval_tasks),
        num_fewshot=num_fewshot
    )

    print('Parent model results: ', results_parent)

    wrapped_model = ModelWrapper(
        model=child_model,
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        device=device
    )

    results_child = evaluator.evaluate(
        lm=wrapped_model,
        task_dict=tasks.get_task_dict(eval_tasks),
        num_fewshot=num_fewshot
    )

    print('Child model results: ', results_child)

    model_path = "./open_llama_3b_v2_first_last_step"
    model = LlamaForCausalLM.from_pretrained(model_path)

    tokenizer=LlamaTokenizer.from_pretrained(
        model_path,
    )

    wrapped_model = ModelWrapper(
        model=model,
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        device=device
    )

    results = evaluator.evaluate(
        lm=wrapped_model,
        task_dict=tasks.get_task_dict(eval_tasks),
        num_fewshot=num_fewshot
    )

    print('Trained child model results: ', results)


if __name__ == "__main__":
    evaluate()
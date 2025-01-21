from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from typing import List
from src.evaluation.model_wrapper import ModelWrapper
from lm_eval import evaluator, tasks
from src.models.prepare_models import prepare_models
from config import evaluate_config, trained_model_config, parent_model_config

BATCH_SIZE = evaluate_config.batch_size

def evaluate():
    eval_tasks: List[str] = evaluate_config.eval_tasks

    tokenizer = LlamaTokenizer.from_pretrained(parent_model_config.pretrained_model_name)
    tokenizer.pad_token_id = 0

    num_fewshot = evaluate_config.num_fewshot
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

    model = LlamaForCausalLM.from_pretrained(trained_model_config.model_path)
    tokenizer=LlamaTokenizer.from_pretrained(trained_model_config.model_path)

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
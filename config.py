from pydantic import BaseModel
from typing import List

class ParentModelConfig(BaseModel):
    pretrained_model_name: str = "openlm-research/open_llama_3b_v2"

class TrainedModelConfig(BaseModel):
    model_path: str = "./open_llama_3b_v2_first_last_step"

class DatasetConfig(BaseModel):
    wikipedia_dataset_name: str = 'wikipedia'
    wikipedia_dataset_version: str = '20220301.en'
    bookcorpus_dataset_name: str = 'bookcorpus'
    dataset_split: str = 'train[:1%]'
    filter_threshold: int = 10


class EvaluateConfig(BaseModel):
    eval_tasks: List[str] = ["winogrande", "boolq", "piqa"]
    batch_size: int = 8
    num_fewshot: int = 0


class WrapperConfig(BaseModel):
    max_toks: int = 256


class PrepareConfig(BaseModel):
    num_hidden_layers: int = 6
    layers: List[int] = [0, 3, 10, 15, 22, 25]

class DistillationLossConfid(BaseModel):
    loss_reduction_name: str = "batchmean"

class TokenizerConfig(BaseModel):
    max_length: int = 2048

class TrainDistillationConfig(BaseModel):
    batch_size: int = 8
    num_epochs: int = 4
    step_log_period: int = 50
    lr: float = 3e-4
    temperature: float = 2.0
    model_path: str = "./distillation_llama"

class TrainInherituneConfig(BaseModel):
    batch_size: int = 8
    num_epochs: int = 6
    per_device_train_batch_size: int = 8
    training_args_output_dir: str = "./results"
    save_steps: float = 10_000
    save_total_limit: int = 2
    gradient_accumulation_steps: int = 8
    dataloader_num_workers: int = 4

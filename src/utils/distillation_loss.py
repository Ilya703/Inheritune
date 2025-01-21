import torch
from config import DistillationLossConfid

'''
    Функция потерь для дистилляции
'''

def distillation_loss(student_logits, teacher_logits, temperature):
    loss_fn = torch.nn.KLDivLoss(reduction=DistillationLossConfid.loss_reduction_name)
    student_probs = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
    
    return loss_fn(student_probs, teacher_probs)
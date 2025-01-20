'''
    Функция для подсчета параметров модели
'''

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters()) / 10**6
    print(f'total_params: {total_params:.3f}M')
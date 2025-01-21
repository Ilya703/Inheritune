'''
    Функция для сжатия датасета
'''

def filter_dataset(dataset, percents):
    part = percents / 100
    return dataset.shuffle().select(range(int(len(dataset)*part)))

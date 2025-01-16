## Pre-training Small Base LMs with Fewer Tokens

Исследование переноса обучения методами дистилляции и Inheritune на основе статьи "Pre-training Small Base LMs with Fewer Tokens"

В исследовании участвовали: Бычкова Марина (БИВТ-21-16), Скобелев Илья (БИВТ-21-17)

Авторы статьи рассказывают про то, как быстро обучить новую небольшую модель, если стартовать с блоков из обученной большой. Статья: https://arxiv.org/html/2404.08634v1, github: https://github.com/sanyalsunny111/LLM-Inheritune

## Проблема и цели исследования

Проблема: В сфере Machine Learning постоянно стоит задача оптимизации использования ресурсов с целью получить наилучшее обучение модели

Цели исследования:

- Обучить модели методом Inheritune, предложенным в статье, и методом дистилляции
- Сравнить точность малых моделей с родительской и между собой
- Понять, в каком случае какой метод использовать

## Метод Inheritune

Авторы исследовали эффективность простого подхода к разработке небольшой базовой модели языка (LM), исходя из существующей большой базовой LM

Они наследовали несколько блоков трансформера от более крупной LM, а затем обучали эту меньшую модель на очень малой выборке (0.1%) исходного датасета для обучения крупной модели. Этот подход назван Inheritune

Полученные результаты были близки к метрикам родительской модели

![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUet9J0lzC43SlsLM72z2KFsD5Sax10Shmvh_jkQ7rIO9nYrgw_lZEds9oqlnsJI6Q8vx2BFbUhUN4ieRZhEACEmBXgLceq5Y7YDv8ZT5xavPpSXLmmyVvCRKM8lrri7GAlhZKJwdDom7ovoR36wHh-rpnM3rIY=s2048?key=d-35EwHvnTAy0h7zxhSG6g)

## Метод дистилляции

Метод дистилляции - это процесс передачи знаний от одной модели к другой, меньшей модели. Это достигается путем обучения ученика имитировать выходные данные учителя или его промежуточные представления

Меньшая модель обучается не только на выходных метках, но и на информации, извлеченной из предсказаний более крупной модели

![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUf_dP8kl-o7eLCCKoqwx2rDXx2D4jMkn7EVgDoJwXFBkIt0nD1sm5XwMhKCGy4zKQHghjhiyKXYddVF6-vSfP5p3Wf_EUjM0euCJuQfHdg0dFb4vvOWsmKiM9MPAXTjViWTO8o74kUvFnyvLKHBoerZCzSjfgN4=s2048?key=d-35EwHvnTAy0h7zxhSG6g)

## Отличие обучения Inheritune от метода дистилляции

В отличие от дистилляции, в методе Inheritune не используется модель-учитель

Меньшая модель обучается на подмножестве данных, используемых для обучения учителя (0.1%). Это позволяет ученику быстро научиться задачам, для которых была обучена большая модель, экономя при этом вычислительные ресурсы

## Как проходило исследование

В рамках исследования мы сравнивали 3 конфигурации “модель + обучение + валидация”. В основе брали модель openlm-research/open_llama_3b_v2

- llama
- llama методом Inheritune
- llama методом дистилляции

## Методы оценки

Для оценки использовали библиотеку lm-eval-harness ([https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)), написали обертку-класс и измеряли zero-shot accuracy

**![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUf2JuZ18M37PurqqslbajZzUQ-IuY2LLTzDlkIOfX7GCLk8vcllewQOlp7uMJp5PrGAPxP4ruv35ByAVYzhoTrQOToC56KuiJOFol9zJQpriCIEaXhA8xzoQ2t0BiUnFOh39lv2usntJBRiJKISW6UJFVHAPPHi=s2048?key=d-35EwHvnTAy0h7zxhSG6g)**

**![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUdoI9Zi8_tfgGzGH3-W_kPnu1eustGI3rrabi00QL2O9tVlASO23Di6haFBakKEzU_GodFX_VSRABQ3Ygm-17x_S9Xj7gnDGLGRKyfkRjF1wMN1HILptd3Q70uacQwxdZeVqlGjN7mcbxpIRat7dUggPbVSsBTq=s2048?key=d-35EwHvnTAy0h7zxhSG6g)**

## Использованные датасеты

Мы взяли 3 датасета из статьи для оценки : boolq, winogrand, piqa. Для обучения брали датасеты wikipedia и bookcorpus (по 0.1%)

![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUeY4VJjbvkdHIOKHEfOaY4EikBL4BzCTAp5NEnZCPAyRS4HseIyLJKyxspFpJzeKTZZ28cDm0hhIxKNqoLhsjn-B42_g9uy18FIlScZt3hT5jNQzi5FP13eBq7En5jR2ghgprueOcFXi34cM0ICFGuTmzd9WsNb=s2048?key=d-35EwHvnTAy0h7zxhSG6g)

## Llama: наследование без обучения

Мы брали первый, последний слои и несколько слоев между ними с определенным шагом, была идея, что так получится лучший результат. В итоге поулчили почти 1 миллиард параметров

Наследовали 6 слоев из 26 в силу ограниченных ресурсов

Обучали модели на Kaggle и Yandex DataSphere

![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUdB2VTQ522iJq-2CTIHD2zcJz3YKdPZaWHI2NPdb6pknFiio5cY9Gil90buQaA0pUfgO04Kz2PRGk6rDfqSdypf17RgMsQ8lX9F6ylHJh3pzYSpiC8STasiSFRhBanctzVuODBpDaJIp7pkQNkk6-jHjBIgQs7D=s2048?key=d-35EwHvnTAy0h7zxhSG6g)

## Inheritune

Взяли 6 блоков: первый, последний и 4 промежуточных между ними

Learning rate: 3e-5

Время обучения: 12 часов, видеокарта A100

Batch size: 8

Epoch: 6

**![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUdzHmL7GTp2fWACJjloeIYXvxPnVzmM9lOVrhCa7x_4EWSEDQBmX9iYQzHpNO7_m14SODF6hY4JBHNBChednfz5v5Bv77VVIQKJ0jrZZiSZbq56Xw8nSOm4J09acrWHQvzqc2RB0EPBB0WhrQB35zjuLsR069gc=s2048?key=d-35EwHvnTAy0h7zxhSG6g)**

## Метод дистилляции

В методе distillation_loss использовали KL дивергенцию, которая учитывает не только таргеты, но и предсказания модели-учителя

**![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUfnARr2SqMMWMMsBwSDbraJpkLAU5TrIhugMq7tMjhRob0eUEUgfb9d9FDm5leBlnBeKZlBSv4OgeGZkX3wABD8aeKijaWzcNTvj9jVyFIIdGE1wFW9lYjh6sPtWiiW8gEcuMnUY6A0JFt705woHSHhAo-Wh2dJ=s2048?key=d-35EwHvnTAy0h7zxhSG6g)**

Взяли 6 блоков: первый, последний и 4 промежуточных между ними

Learning rate: 3e-5

Время обучения: 72 часов, видеокарта A100

Batch size: 8

**![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUf-zsZsaq8Sj2QENX7wEAMKoI9Khcj8FCwHzt0FUezTCnY6p_Goo-AiQivcWfFPA7cVqkecxCtO52_j3IXG20JxwF3sh7kj7lSWUGfIFlI3Hbrn3-8z95sbgeXUPVQ5DyjtqiR17eY-azZb8dI6xUmevZpusk78=s2048?key=d-35EwHvnTAy0h7zxhSG6g)**

Использовали 5 эпох, при которых обучение занимало 72 часа. В силу ограниченности ресурсов не удалось до конца обучить модели методом дистилляции. Пришли к выводу, что в положении, когда ограничены время и русурсы, метод Inheritune значительно опережает метод дистилляции

## Результаты исследования

Результаты обучения методом Inheritune:

- зеленый - показатели родительской модели
- синий - показатели дочерней модели до обучения (отнаследовали 6 слоев и оценили)
- оранжевый - показатели дочерней модели после обучения на 0.1% датасета

Для датасетов winogrande и piqa точность выросла, а для boolq немного упала. Это может быть связано с набором используемых датасетов (wikipedia и bookcorpus), которые использовали в силу ограничений по памяти и времени. При более широком спектре датасетов результаты на датасете boolq скорее всего так же бы увеличились

Точность дочерней модели до обучения на некоторых датасетах оказалась выше, чем в статье. Это связано с тем, что мы не как авторы наследовали первые n блоков, а брали первый, последний и несколько промежуточных с определенным шагом

**![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUfn1YukkhLDOwMXMBnI6w3hlmFTTcG_Gv3AxMhyany1La4C0h8AhSpoIskqELVlWHYkvER32t-lNVXGRjCFQWl675i8DiHogMdugrCLv2W2Kb5H7x11E_UqziLO5cds1-ju-bsgEuP9NpIfMv3SczvbfPlhAAmv=s2048?key=d-35EwHvnTAy0h7zxhSG6g)**

## Выводы: преимущества и недостатки

Метод Inheritune:

- Преимущества: быстрое обучение
- Недостатки: зависит от качества и объема данных

Метод дистилляции:

- Недостатки: высокие вычислительные затраты

Таким образом, в случае ограничения по времени и ресурсам метод Inheritune становится отличным простым методом для сжатия больших моделей

MLOps Project
=============

Проект реализует ML-конвейер для обработки потоковых данных.

Структура проекта
-----------------
::

     MLOpsTask
     ├── src
     │   ├── data_analyzer.py      # Анализ данных
     │   ├── data_collector.py     # Хранение данных
     │   ├── data_provider.py      # Эмуляция потока
     │   ├── data_transformer.py   # Подготовка данных для ML-модели
     │   ├── model.py              # ML-модель
     │   └── utils.py              # Вспомогательные функции
     ├── main.py            # Входная точка
     ├── Pipfile            # Pipenv-конфигурация
     ├── requirements.txt   # Зависимости
     └── README.rst         # Документация

..

Использование
-------------
Обучение и выбор наилучшей модели: ::

    python3 main.py --mode train --data <path_to_dataset> [--n_iter <int>] [--verbose]

..

Обучение модели на новых данных: ::

    python3 main.py --mode update --data <path_to_dataset> [--verbose]

..

Оценка качества модели: ::

    pythonn3 main.py --mode eval --data <path_to_dataset> [--n_iter <int>] [--verbose]

..

Применение модели: ::

    python3 main.py --mode inference --data <path_to_dataset> --out <path_to_output_file> [--verbose]

..

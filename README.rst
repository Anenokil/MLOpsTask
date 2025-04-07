MLOps Project
=============

The project implements ML Pipeline to handle stream data.

Project structure
-----------------
::

     MLOpsTask
     ├── src
     │   ├── data_analyzer.py      # Analyzes data
     │   ├── data_collector.py     # Stores data
     │   ├── data_provider.py      # Emulates data stream
     │   ├── data_transformer.py   # Prepare data for model
     │   ├── model.py              # Manages the training process
     │   └── utils.py              # Auxiliary functions
     ├── main.py            # Entry point of the application
     ├── Pipfile            # Pipenv configuration
     ├── requirements.txt   # Project dependencies
     └── README.md          # Project documentation

..

Usage
-----
Train: ::

    python3 main.py --mode train -data <path_to_dataset> [--verbose]

..

Inference: ::

    python3 main.py --mode inference -data <path_to_dataset> -o <path_to_output_file> [--verbose]

..

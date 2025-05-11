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
     │   ├── data_transformer.py   # Prepares data for model
     │   ├── model.py              # Manages the training process
     │   └── utils.py              # Auxiliary functions
     ├── main.py            # Entry point of the application
     ├── Pipfile            # Pipenv configuration
     ├── requirements.txt   # Project dependencies
     └── README.rst         # Project documentation

..

Usage
-----
Usage example: ::

    # Training
    python3 main.py --mode train --data <path_to_dataset> [--n_iter <int>] [--verbose]

    # Evaluation
    pythonn3 main.py --mode eval --data <path_to_dataset> [--n_iter <int>] [--verbose]

    # Inference
    python3 main.py --mode inference --data <path_to_dataset> --out <path_to_output_file> [--verbose]

..

CI/CD
-----
There is train.yaml workflow, which is triggered by push/pull request.
Steps:
- Install requirements;
- Train model;
- Save logs.

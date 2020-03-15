"""
AI-Utilities - train_local.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.


coding: utf-8

# Train Locally
In this notebook, you will perform the following using Azure Machine Learning.
* Load workspace.
* Configure & execute a local run in a user-managed Python environment.
* Configure & execute a local run in a system-managed Python environment.
* Configure & execute a local run in a Docker environment.
# * Register model for operationalization.
"""

import os
import sys

from azureml.core.runconfig import RunConfiguration


def get_or_create_model_driver(train_py: str = "create_model.py"):
    """ Create Model Script for LightGBM with Stack Overflow Data """
    if not os.path.isfile(f"script/{train_py}"):
        os.makedirs("script", exist_ok=True)

        create_model_py = (
            "from azure_utils.machine_learning import create_model\n\nif __name__ == '__main__':\n    "
            "create_model.main() "
        )
        with open(train_py, "w") as file:
            file.write(create_model_py)


def get_local_run_configuration() -> RunConfiguration:
    """
    Get Local Run Config

    :return:
    """
    # Editing a run configuration property on-fly.
    run_config_user_managed = RunConfiguration()
    run_config_user_managed.environment.python.user_managed_dependencies = True
    # Choose the specific Python environment of this tutorial by pointing to the Python path
    run_config_user_managed.environment.python.interpreter_path = sys.executable
    return run_config_user_managed

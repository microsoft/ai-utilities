"""
AI-Utilities - train_local.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

# coding: utf-8

# Copyright (c) Microsoft Corporation. All rights reserved.
#
# Licensed under the MIT License.
#
# # Train Locally
# In this notebook, you will perform the following using Azure Machine Learning.
# * Load workspace.
# * Configure & execute a local run in a user-managed Python environment.
# * Configure & execute a local run in a system-managed Python environment.
# * Configure & execute a local run in a Docker environment.
# * Register model for operationalization.

# In[19]:


import os
import sys

from azureml.core import Experiment
from azureml.core import ScriptRunConfig
from azureml.core.runconfig import RunConfiguration

from azure_utils import directory
from azure_utils.machine_learning.realtime.image import get_model, has_model
from azure_utils.machine_learning.utils import get_workspace_from_config


def train_local(model_name="question_match_model", num_estimators="1", experiment_name="mlaks-train-on-local",
                model_path="./outputs/model.pkl", script="create_model.py", source_directory="./script",
                show_output=True):
    model = has_model(model_name)
    if model:
        return get_model(model_name)

    ws = get_workspace_from_config()
    if show_output:
        print(ws.name, ws.resource_group, ws.location, sep="\n")

    args = [
        "--inputs",
        os.path.abspath(directory + "/data_folder"),
        "--outputs",
        "outputs",
        "--estimators",
        num_estimators,
        "--match",
        "2",
    ]
    run_config_user_managed = get_run_configuration()

    src = ScriptRunConfig(
        source_directory=source_directory,
        script=script,
        arguments=args,
        run_config=run_config_user_managed,
    )

    run = Experiment(workspace=ws, name=experiment_name).submit(src)
    run.wait_for_completion(show_output=show_output)

    model = run.register_model(model_name=model_name, model_path=model_path)

    if show_output:
        print(model.name, model.version, model.url, sep="\n")
    return model


def create_stack_overflow_model_script():
    if not os.path.isfile("script/create_model.py"):
        os.makedirs("script", exist_ok=True)

        create_model_py = "from azure_utils.machine_learning import create_model\n\nif __name__ == '__main__':\n    " \
                          "create_model.main()"
        with open("script/create_model.py", "w") as file:
            file.write(create_model_py)


def get_run_configuration():
    # Editing a run configuration property on-fly.
    run_config_user_managed = RunConfiguration()
    run_config_user_managed.environment.python.user_managed_dependencies = True
    # Choose the specific Python environment of this tutorial by pointing to the Python path
    run_config_user_managed.environment.python.interpreter_path = sys.executable
    return run_config_user_managed

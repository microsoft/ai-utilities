"""
AI-Utilities - azureml_tools/config.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import ast
import logging

from dotenv import dotenv_values, find_dotenv, set_key

defaults = {
    "CLUSTER_NAME": "gpucluster24rv3",
    "CLUSTER_VM_SIZE": "Standard_NC24rs_v3",
    "CLUSTER_MIN_NODES": 0,
    "CLUSTER_MAX_NODES": 2,
    "WORKSPACE": "workspace",
    "RESOURCE_GROUP": "amlccrg",
    "REGION": "eastus",
    "DATASTORE_NAME": "datastore",
    "CONTAINER_NAME": "container",
    "ACCOUNT_NAME": "premiumstorage",
    "SUBSCRIPTION_ID": None,
}


def load_config(dot_env_path: find_dotenv(raise_error_if_not_found=True)):
    """ Load the variables from the .env file
    Returns:
        .env variables(dict)
    """
    logger = logging.getLogger(__name__)
    logger.info("Found config in %s" % dot_env_path)
    return dotenv_values(dot_env_path)


def _convert(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


class AzureMLConfig:
    """Creates AzureMLConfig object

    Stores all the configuration options and syncs them with the .env file
    """

    _reserved = ("_dot_env_path",)

    def __init__(self):
        self._dot_env_path = find_dotenv(raise_error_if_not_found=True)

        for key, value in load_config(dot_env_path=self._dot_env_path).items():
            self.__dict__[key] = _convert(value)

        for key, value in defaults.items():
            if key not in self.__dict__:
                setattr(self, key, value)

    def __setattr__(self, name, value):
        if name not in self._reserved:
            if not isinstance(value, str):
                value = str(value)
            set_key(self._dot_env_path, name, value)
        self.__dict__[name] = value


experiment_config = AzureMLConfig()

"""
AI-Utilities - ${FILE_NAME}

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from azureml.core import Model

from azure_utils.configuration.notebook_config import project_configuration_file
from azure_utils.configuration.project_configuration import ProjectConfiguration
from azure_utils.machine_learning.utils import get_or_create_workspace_from_project


def get_model(model_name, configuration_file=project_configuration_file, show_output=True):
    """

    :param model_name:
    :param configuration_file:
    :param show_output:
    :return:
    """
    project_configuration = ProjectConfiguration(configuration_file)
    workspace = get_or_create_workspace_from_project(project_configuration, show_output=show_output)
    return Model(workspace, name=model_name)


def has_model(model_name, configuration_file=project_configuration_file, show_output=True):
    """

    :param model_name:
    :param configuration_file:
    :param show_output:
    :return:
    """
    project_configuration = ProjectConfiguration(configuration_file)
    workspace = get_or_create_workspace_from_project(project_configuration, show_output=show_output)
    return model_name in workspace.models



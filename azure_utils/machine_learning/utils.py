"""
ado-ml-batch-train - machine_learning/utils.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import json
import os
from typing import Union

import yaml
import azureml.core
from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication, ServicePrincipalAuthentication
from deprecated import deprecated

from azure_utils import directory
from azure_utils.configuration.notebook_config import project_configuration_file
from azure_utils.configuration.project_configuration import ProjectConfiguration


@deprecated(version='0.2.8', reason="Switch to using ProjectConfiguration, this will be removed in 0.4.0")
def load_configuration(configuration_file: str):
    """
    Load the Workspace Configuration File.

    The workspace configuration file is used to protect against putting passwords within the code, or source control.
    To create the configuration file, make a copy of sample_workspace.conf named "workspace_conf.yml" and fill in
    each field.
    This file is set to in the .gitignore to prevent accidental comments.

    :param configuration_file: File Path to configuration yml
    :return: Returns the parameters needed to configure the AML Workspace and Experiments
    :rtype: Union[Dict[Hashable, Any], list, None], str, str, str, str, Workspace, str, str
    """
    if not os.path.isfile(configuration_file):
        configuration_file = directory + "/../sample_workspace_conf.yml"

    with open(configuration_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    return cfg


def get_or_create_workspace(workspace_name: str, subscription_id: str, resource_group: str, workspace_region: str,
                            auth: Union[InteractiveLoginAuthentication, ServicePrincipalAuthentication] =
                            InteractiveLoginAuthentication(), log=True) -> Workspace:
    """
    Create a new Azure Machine Learning workspace. If the workspace already exists, the existing workspace will be
    returned. Also create a CONFIG file to quickly reload the workspace.

    This uses the :class:`azureml.core.authentication.InteractiveLoginAuthentication` or will default to use the
    :class:`azureml.core.authentication.AzureCliAuthentication` for logging into Azure.

    Run az login from the CLI in the project directory to avoid authentication when running the program.

    :param workspace_name: Name of Azure Machine Learning Workspace to get or create within the Azure Subscription
    :type workspace_name: str
    :param subscription_id: Azure Subscription ID
    :type subscription_id: str
    :param resource_group: Azure Resource Group to get or create the workspace within. If the resource group does not
    exist it will be created.
    :type resource_group: str
    :param workspace_region: The Azure region to deploy the workspace.
    :type workspace_region: str
    :param auth: Derived classes provide different means to authenticate and acquire a token based on their targeted
    use case.
    For examples of authentication, see https://aka.ms/aml-notebook-auth.
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param log: enable print output
    :return: Returns a :class:`azureml.core.Workspace` object, a pointer to Azure Machine Learning Workspace
    Learning Workspace
    :rtype: azureml.core.Workspace
    """
    if log:
        print("AML SDK Version:", azureml.core.VERSION)

    workspace = Workspace.create(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group,
                                 location=workspace_region, create_resource_group=True, auth=auth, exist_ok=True)

    workspace.write_config()

    if log:
        ws_json = json.dumps(workspace.get_details(), indent=2)
        print(ws_json)

    return workspace


def get_or_create_workspace_from_project(project_configuration: ProjectConfiguration,
                                         auth: Union[InteractiveLoginAuthentication, ServicePrincipalAuthentication] =
                                         InteractiveLoginAuthentication(), show_output=True) -> Workspace:
    """
    Create a new Azure Machine Learning workspace. If the workspace already exists, the existing workspace will be
    returned. Also create a CONFIG file to quickly reload the workspace.

    This uses the :class:`azureml.core.authentication.InteractiveLoginAuthentication` or will default to use the

    :class:`azureml.core.authentication.AzureCliAuthentication` for logging into Azure.

    Run az login from the CLI in the project directory to avoid authentication when running the program.

    :param project_configuration: Project Configuration Container
    :param auth: Derived classes provide different means to authenticate and acquire a token based on their targeted
    use case.
    For examples of authentication, see https://aka.ms/aml-notebook-auth.
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param show_output: enable print output
    :return: Returns a :class:`azureml.core.Workspace` object, a pointer to Azure Machine Learning Workspace
    Learning Workspace
    """
    return get_or_create_workspace(project_configuration.get_value('workspace_name'),
                                   project_configuration.get_value('subscription_id'),
                                   project_configuration.get_value('resource_group'),
                                   project_configuration.get_value('workspace_region'),
                                   auth=auth, log=show_output)


def get_or_create_workspace_from_file(configuration_file: str = project_configuration_file,
                                      auth: Union[InteractiveLoginAuthentication, ServicePrincipalAuthentication] =
                                      InteractiveLoginAuthentication(), log=True) -> Workspace:
    """
    Create a new Azure Machine Learning workspace. If the workspace already exists, the existing workspace will be
    returned. Also create a CONFIG file to quickly reload the workspace.

    This uses the :class:`azureml.core.authentication.InteractiveLoginAuthentication` or will default to use the

    :class:`azureml.core.authentication.AzureCliAuthentication` for logging into Azure.

    Run az login from the CLI in the project directory to avoid authentication when running the program.

    :param configuration_file: File path to project configuration file. default: ../project.yml
    :param auth: Derived classes provide different means to authenticate and acquire a token based on their targeted
    use case.
    For examples of authentication, see https://aka.ms/aml-notebook-auth.
    :param log: enable print output
    :type auth: azureml.core.authentication.AbstractAuthentication
    :return: Returns a :class:`azureml.core.Workspace` object, a pointer to Azure Machine Learning Workspace
    Learning Workspace
    """
    project_configuration = ProjectConfiguration(configuration_file)

    return get_or_create_workspace(project_configuration.get_value('workspace_name'),
                                   project_configuration.get_value('subscription_id'),
                                   project_configuration.get_value('resource_group'),
                                   project_configuration.get_value('workspace_region'),
                                   auth=auth, log=log)


def get_workspace_from_config():
    """
    Retrieve an AML Workspace from a previously saved configuration

    :return: Azure Machine Learning Workspace
    :rtype: azureml.core.Workspace
    """
    return Workspace.from_config()

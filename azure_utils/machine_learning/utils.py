"""
ado-ml-batch-train - machine_learning/utils.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import os
from typing import Union

import yaml
from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication, ServicePrincipalAuthentication


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
        configuration_file = "../sample_workspace_conf.yml"

    with open(configuration_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    return cfg


def get_or_create_workspace(workspace_name: str, subscription_id: str, resource_group: str, workspace_region: str,
                            auth: Union[InteractiveLoginAuthentication, ServicePrincipalAuthentication] =
                            InteractiveLoginAuthentication()) -> Workspace:
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
    :return: Returns a :class:`azureml.core.Workspace` object, a pointer to Azure Machine Learning Workspace
    Learning Workspace
    :rtype: azureml.core.Workspace
    """
    workspace = Workspace.create(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group,
                                 location=workspace_region, create_resource_group=True, auth=auth, exist_ok=True)
    workspace.write_config()

    return workspace


def get_workspace_from_config():
    """
    Retrieve an AML Workspace from a previously saved configuration

    :return: Azure Machine Learning Workspace
    :rtype: azureml.core.Workspace
    """
    return Workspace.from_config()

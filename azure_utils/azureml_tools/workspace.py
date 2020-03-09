"""
AI-Utilities - workspace

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import logging
import os
from pathlib import Path

import azureml
from azureml.core.authentication import (
    AuthenticationException,
    AzureCliAuthentication,
    InteractiveLoginAuthentication,
    ServicePrincipalAuthentication,
    AbstractAuthentication,
)
from azureml.core import Workspace

_DEFAULT_AML_PATH = "aml_config/azml_config.json"


def _get_auth() -> AbstractAuthentication:
    """Returns authentication to Azure Machine Learning workspace."""
    logger = logging.getLogger(__name__)
    if os.environ.get("AML_SP_PASSWORD", None):
        logger.debug("Trying to authenticate with Service Principal")
        aml_sp_password = os.environ.get("AML_SP_PASSWORD")
        aml_sp_tenant_id = os.environ.get("AML_SP_TENNANT_ID")
        aml_sp_username = os.environ.get("AML_SP_USERNAME")
        auth = ServicePrincipalAuthentication(
            aml_sp_tenant_id, aml_sp_username, aml_sp_password
        )
    else:
        logger.debug("Trying to authenticate with CLI Authentication")
        try:
            auth = AzureCliAuthentication()
            auth.get_authentication_header()
        except AuthenticationException:
            logger.debug("Trying to authenticate with Interactive login")
            auth = InteractiveLoginAuthentication()

    return auth


def create_workspace(
    workspace_name: str,
    resource_group: str,
    subscription_id: str,
    workspace_region: str,
    filename: str = "azml_config.json",
) -> Workspace:
    """Creates Azure Machine Learning workspace."""
    logger = logging.getLogger(__name__)
    auth = _get_auth()

    # noinspection PyTypeChecker
    ws = azureml.core.Workspace.create(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        location=workspace_region,
        exist_ok=True,
        auth=auth,
    )

    logger.info(ws.get_details())
    ws.write_config(file_name=filename)
    return ws


def load_workspace(path: str) -> Workspace:
    """Loads Azure Machine Learning workspace from a config file."""
    auth = _get_auth()
    # noinspection PyTypeChecker
    workspace = azureml.core.Workspace.from_config(auth=auth, path=path)
    logger = logging.getLogger(__name__)
    logger.info(
        "\n".join(
            [
                "Workspace name: " + str(workspace.name),
                "Azure region: " + str(workspace.location),
                "Subscription id: " + str(workspace.subscription_id),
                "Resource group: " + str(workspace.resource_group),
            ]
        )
    )
    return workspace


def workspace_for_user(
    workspace_name: str,
    resource_group: str,
    subscription_id: str,
    workspace_region: str,
    config_path: str = _DEFAULT_AML_PATH,
) -> Workspace:
    """Returns Azure Machine Learning workspace."""
    if os.path.isfile(config_path):
        return load_workspace(config_path)

    path_obj = Path(config_path)
    filename = path_obj.name
    return create_workspace(
        workspace_name,
        resource_group,
        subscription_id=subscription_id,
        workspace_region=workspace_region,
        filename=filename,
    )

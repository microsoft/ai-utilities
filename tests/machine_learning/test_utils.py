"""
ai-utilities - test_utils.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import os

import pytest
from azureml.core import Workspace

from azure_utils import directory
from azure_utils.machine_learning.contexts.workspace_contexts import WorkspaceContext
from azure_utils.machine_learning.utils import get_or_create_workspace, get_or_create_workspace_from_file, \
    get_workspace_from_config, load_configuration

filepath = directory


# def test_load_configuration():
#     """Test Loading Configuration to check sample file contents"""
#
#     cfg = load_configuration(filepath + "/../sample_workspace_conf.yml")
#     assert cfg
#
#     assert cfg['subscription_id'] == '<>'
#     assert cfg['resource_group'] == '<>'
#     assert cfg['workspace_name'] == '<>'
#     assert cfg['workspace_region'] == '<>'
#     assert cfg['image_name'] == '<>'
#
#     assert cfg['sql_server_name'] == '<>'
#     assert cfg['sql_database_name'] == '<>'
#     assert cfg['sql_username'] == '<>'
#     assert cfg['sql_password'] == '<>'
#
#     assert cfg['datastore_rg'] == '<>'
#     assert cfg['container_name'] == '<>'
#     assert cfg['account_name'] == '<>'
#     assert cfg['account_key'] == '<>'


# def test_get_or_create_workspace(workspace):
#     """Test Get or Create Workspace Method"""
#     assert isinstance(workspace, Workspace)
#     assert os.path.isfile('./.azureml/config.json')


def dont_test_get_workspace_from_config():
    """ Test Get Workspace From Config File"""

    workspace = get_workspace_from_config()
    assert isinstance(workspace, Workspace)


def dont_test_get_workspace_from_project_config():
    """ Test Get Workspace From Project File"""

    get_or_create_workspace_from_file()

    workspace = WorkspaceContext.get_or_create_workspace()
    assert isinstance(workspace, Workspace)


# @pytest.fixture
# def cfg():
#     """
#
#     :return:
#     """
#     cfg = load_configuration(filepath + "/../workspace_conf.yml")
#     return cfg


@pytest.fixture
def workspace(cfg):
    """

    :param cfg:
    :return:
    """
    return get_or_create_workspace(cfg['workspace_name'], cfg['subscription_id'], cfg['resource_group'],
                                   cfg['workspace_region'])

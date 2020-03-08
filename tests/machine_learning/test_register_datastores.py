# """
# ai-utilities - test_register_datastores.py
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# """
# import pytest
# from msrest.exceptions import HttpOperationError
#
# from azure_utils import directory
# from azure_utils.machine_learning.register_datastores import register_blob_datastore, register_sql_datastore
# from azure_utils.machine_learning.utils import get_or_create_workspace, load_configuration
#
#
# @pytest.fixture
# def init_test_vars():
#     """
#     Load Common Vars for Testing
#
#     :return: CONFIG, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME, WORKSPACE_REGION, workspace, SQL_DATASTORE_NAME,
#     BLOB_DATASTORE_NAME
#     :rtype: Union[Dict[Hashable, Any], list, None], str, str, str, str, Workspace, str, str
#     """
#     cfg = load_configuration(directory + "/../workspace_conf.yml")
#
#     subscription_id = cfg['subscription_id']
#     resource_group = cfg['resource_group']
#     workspace_name = cfg['workspace_name']
#     workspace_region = cfg['workspace_region']
#
#     workspace = get_or_create_workspace(workspace_name, subscription_id, resource_group, workspace_region)
#
#     sql_datastore_name = "ado_sql_datastore"
#     blob_datastore_name = "ado_blob_datastore"
#
#     return {'cfg': cfg, 'subscription_id': subscription_id, "resource_group": resource_group,
#             "workspace_name": workspace_name, "workspace_region": workspace_region, "workspace": workspace,
#             "sql_datastore_name": sql_datastore_name, "blob_datastore_name": blob_datastore_name}
#
#
# def test_register_blob_datastore(init_test_vars):
#     """ Test Register Blob Datastore Method """
#     datastore_rg = init_test_vars['cfg']['datastore_rg']
#     container_name = init_test_vars['cfg']['container_name']  # Name of Azure blob container
#     account_name = init_test_vars['cfg']['account_name']  # Storage account name
#     account_key = init_test_vars['cfg']['account_key']  # Storage account key
#
#     blob_datastore = register_blob_datastore(init_test_vars['workspace'], init_test_vars['blob_datastore_name'],
#                                              container_name, account_name, account_key, datastore_rg)
#
#     assert blob_datastore
#
#
# def test_register_sql_datastore(init_test_vars):
#     """ Test Register SQL Datatstore Method """
#     sql_server_name = init_test_vars['cfg']['sql_server_name']  # Name of Azure SQL server
#     sql_database_name = init_test_vars['cfg']['sql_database_name']  # Name of Azure SQL database
#     sql_username = init_test_vars['cfg']['sql_username']  # The username of the database user to access the database.
#     sql_password = init_test_vars['cfg']['sql_password']  # The password of the database user to access the database.
#
#     try:
#         sql_datastore = register_sql_datastore(init_test_vars['workspace'], init_test_vars['blob_datastore_name'],
#                                                sql_server_name, sql_database_name, sql_username, sql_password)
#
#         assert sql_datastore
#     except HttpOperationError:
#         pass

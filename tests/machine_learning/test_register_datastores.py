"""
ai-utilities - test_register_datastores.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from azure_utils.machine_learning.register_datastores import register_sql_datastore, register_blob_datastore
from tests.utils import init_test_vars

CONFIG, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME, WORKSPACE_REGION, WORKSPACE, SQL_DATASTORE_NAME, \
BLOB_DATASTORE_NAME = init_test_vars()


def test_register_blob_datastore():
    """ Test Register Blob Datastore Method """
    datastore_rg = CONFIG['datastore_rg']
    container_name = CONFIG['container_name']  # Name of Azure blob container
    account_name = CONFIG['account_name']  # Storage account name
    account_key = CONFIG['account_key']  # Storage account key

    blob_datastore = register_blob_datastore(WORKSPACE, BLOB_DATASTORE_NAME, container_name, account_name, account_key,
                                             datastore_rg)

    assert blob_datastore


def test_register_sql_datastore():
    """ Test Register SQL Datatstore Method """
    sql_server_name = CONFIG['sql_server_name']  # Name of Azure SQL server
    sql_database_name = CONFIG['sql_database_name']  # Name of Azure SQL database
    sql_username = CONFIG['sql_username']  # The username of the database user to access the database.
    sql_password = CONFIG['sql_password']  # The password of the database user to access the database.

    sql_datastore = register_sql_datastore(WORKSPACE, SQL_DATASTORE_NAME, sql_server_name, sql_database_name,
                                           sql_username, sql_password)

    assert sql_datastore

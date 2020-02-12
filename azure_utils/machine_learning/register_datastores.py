"""
ado-ml-batch-train - machine_learning/register_datastores.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from azureml.core import Datastore, Workspace
from azureml.data.azure_sql_database_datastore import AzureSqlDatabaseDatastore
from azureml.data.azure_storage_datastore import AzureBlobDatastore


def register_blob_datastore(workspace: Workspace, blob_datastore_name: str, container_name: str, account_name: str,
                            account_key: str, datastore_rg: str) -> AzureBlobDatastore:
    """
    Register a Blob Storage Account with the Azure Machine Learning Workspace

    :param workspace: Azure Machine Learning Workspace
    :param blob_datastore_name: Name for blob datastore
    :param container_name: Name for blob container
    :param account_name: Name for blob account
    :param account_key: Blob Account Key using for auth
    :param datastore_rg: Resource Group containing Azure Storage Account
    :return: Pointer to Azure Machine Learning Blob Datastore
    """
    return Datastore.register_azure_blob_container(workspace=workspace, datastore_name=blob_datastore_name,
                                                   container_name=container_name, account_name=account_name,
                                                   account_key=account_key, resource_group=datastore_rg, overwrite=True)


def register_sql_datastore(workspace: Workspace, sql_datastore_name: str, sql_server_name: str, sql_database_name: str,
                           sql_username: str, sql_password: str) -> AzureSqlDatabaseDatastore:
    """
    Register a Azure SQL DB with the Azure Machine Learning Workspace

    :param workspace: Azure Machine Learning Workspace
    :param sql_datastore_name: Name used to id the SQL Datastore
    :param sql_server_name: Azure SQL Server Name
    :param sql_database_name: Azure SQL Database Name
    :param sql_username: Azure SQL Database Username
    :param sql_password: Azure SQL Database Password
    :return: Pointer to Azure Machine Learning SQL Datastore
    """
    return Datastore.register_azure_sql_database(workspace=workspace, datastore_name=sql_datastore_name,
                                                 server_name=sql_server_name, database_name=sql_database_name,
                                                 username=sql_username, password=sql_password)

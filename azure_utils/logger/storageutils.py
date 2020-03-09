"""
AI-Utilities - storageutils.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

ImportError: You need to install 'azure-cli-core' to load CLI active Cloud

    REQUIREMENT : pip install azure-cli-core

    Class that parses out a true connection string from Azure Storage account in the form:

    DefaultEndpointsProtocol=https;AccountName=ACCT_NAME;AccountKey=ACCT_KEY;EndpointSuffix=core.windows.net

    Ends up with 4 attributes :
        DefaultEndpointsProtocol
        AccountName
        AccountKey
        EndpointSuffix
"""
from azure.common.client_factory import get_client_from_cli_profile
from azure.mgmt.storage import StorageManagementClient


class StorageConnection:
    """
        Constructor taking the connection string to parse.

        EX:
        DefaultEndpointsProtocol=https;AccountName=STGACCT_NAME;AccountKey=STGACCT_KEY;EndpointSuffix=core.windows.net

    """

    def __init__(self, connection_string):
        parsed_connection_string = self._parse_connection_string(connection_string)
        for key, value in parsed_connection_string.items():
            self.__setattr__(key, value)

    """
        Expects the full connection string from the Azure site and spits it into four components.

        EX:
        DefaultEndpointsProtocol=https;AccountName=STGACCT_NAME;AccountKey=STGACCT_KEY;EndpointSuffix=core.windows.net
    """

    @staticmethod
    def _parse_connection_string(connection_string):
        return_value = {}
        if connection_string:
            segments = connection_string.split(";")
            for segment in segments:
                split_index = segment.index("=")
                second_part = (len(segment) - split_index - 1) * -1
                return_value[segment[:split_index]] = segment[second_part:]

        return return_value

    # Method to return the full connection string to an Azure Storage account give the resource group name and
    # storage account
    # name.
    #
    # Method expects that the environment has been logged into Azure and the subscription has been set to match the
    # incoming
    # resource group and storage account.

    @staticmethod
    def get_connection_string_with_az_credentials(
        resource_group_name, storage_account_name
    ):
        """

        :param resource_group_name:
        :param storage_account_name:
        :return:
        """
        connection_string_template = (
            "DefaultEndpointsProtocol=https;AccountName={};AccountKey={"
            "};EndpointSuffix=core.windows.net"
        )
        return_value = None

        client = get_client_from_cli_profile(StorageManagementClient)
        keys = client.storage_accounts.list_keys(
            resource_group_name, storage_account_name
        )
        key_value = None
        for key in keys.keys:
            key_value = key.value
            break

        if key_value is not None:
            return_value = connection_string_template.format(
                storage_account_name, key_value
            )

        return return_value

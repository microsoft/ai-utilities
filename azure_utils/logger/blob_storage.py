"""
AI-Utilities - MetricUtils/blobStorage.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Class that performs work against an Azure Storage account with containers and blobs.

To use, this library must be installed:

pip install azure-storage-blob
"""

from datetime import datetime, timedelta

from azure.storage.blob import BlobPermissions, BlockBlobService, PublicAccess


class BlobStorageAccount:
    """
        Constructor that receives a logger.storageutils.storageConnection instance
    """

    def __init__(self, storage_connection):
        self.connection = storage_connection
        self.service = BlockBlobService(
            self.connection.AccountName, self.connection.AccountKey
        )

    # Creates a new storage container in the Azure Storage account

    def create_container(self, container_name):
        """

        :param container_name:
        """
        if self.connection and self.service:
            self.service.create_container(container_name)
            self.service.set_container_acl(
                container_name, public_access=PublicAccess.Blob
            )

    # Retrieve a blob SAS token on a specific blob

    def get_blob_sas_token(self, container_name, blob_name):
        """

        :param container_name:
        :param blob_name:
        :return:
        """
        return_token = None
        if self.connection and self.service:
            # noinspection PyUnresolvedReferences,PyTypeChecker
            return_token = self.service.generate_blob_shared_access_signature(
                container_name,
                blob_name,
                BlobPermissions.READ,
                datetime.utcnow() + timedelta(hours=1),
            )

        return return_token

    # Retrieves a list of storage container names in the specific storage account pointed to by
    # the storageConnection object

    def get_containers(self):
        """

        :return:
        """
        return_list = []
        if self.connection and self.service:
            containers = self.service.list_containers()
            for container in containers:
                return_list.append(container.name)

        return return_list

    # Retrieves a list of storage blob names in a container in the specific storage account pointed to by
    # the storageConnection object

    def get_blobs(self, container_name):
        """

        :param container_name:
        :return:
        """
        return_list = []
        if self.connection and self.service:
            blobs = self.service.list_blobs(container_name)
            for blob in blobs:
                return_list.append(blob.name)
        return return_list

    # Upload text to a blob (fileContent is a simple string)

    def upload_blob(self, container_name, blob_name, file_content):
        """

        :param container_name:
        :param blob_name:
        :param file_content:
        """
        if self.connection and self.service:
            self.service.create_blob_from_text(container_name, blob_name, file_content)

    # Download the blob as a string.

    def download_blob(self, container_name, blob_name):
        """

        :param container_name:
        :param blob_name:
        :return:
        """
        return_content = None
        if self.connection and self.service:
            blob = self.service.get_blob_to_text(container_name, blob_name)
            return_content = blob.content
        return return_content

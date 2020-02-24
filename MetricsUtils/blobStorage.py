"""
AI-Utilities - MetricUtils/blobStorage.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Class that performs work against an Azure Storage account with containers and blobs.

To use, this library must be installed:

pip install azure-storage-blob
"""

from datetime import datetime, timedelta

from azure.storage.blob import BlockBlobService, PublicAccess, BlobPermissions

__version__ = "0.1"


class blobStorageAccount:
    '''
        Constructor that recieves a MetricsUtils.storagutils.storageConnection instance
    '''

    def __init__(self, storageConnection):
        self.connection = storageConnection
        self.service = BlockBlobService(self.connection.AccountName, self.connection.AccountKey)

    '''
        Creates a new storage container in the Azure Storage account
    '''

    def createContainer(self, containerName):
        if self.connection and self.service:
            self.service.create_container(containerName)
            self.service.set_container_acl(containerName, public_access=PublicAccess.Blob)

    '''
        Retrieve a blob SAS token on a specific blob
    '''

    def getBlobSasToken(self, containerName, blobName):
        returnToken = None
        if self.connection and self.service:
            returnToken = self.service.generate_blob_shared_access_signature(containerName, blobName,
                                                                             BlobPermissions.READ,
                                                                             datetime.utcnow() + timedelta(hours=1))

        return returnToken

    '''
        Retrieves a list of storage container names in the specific storage account pointed to by
        the storageConnection object
    '''

    def getContainers(self):
        returnList = []
        if self.connection and self.service:
            containers = self.service.list_containers()
            for container in containers:
                returnList.append(container.name)

        return returnList

    '''
        Retrieves a list of storage blob names in a container in the specific storage account pointed to by
        the storageConnection object
    '''

    def getBlobs(self, containerName):
        returnList = []
        if self.connection and self.service:
            blobs = self.service.list_blobs(containerName)
            for blob in blobs:
                returnList.append(blob.name)
        return returnList

    '''
        Upload text to a blob (fileContent is a simple string)
    '''

    def uploadBlob(self, containerName, blobName, fileContent):
        if self.connection and self.service:
            self.service.create_blob_from_text(containerName, blobName, fileContent)

    '''
        Download the blob as a string.
    '''

    def downloadBlob(self, containerName, blobName):
        returnContent = None
        if self.connection and self.service:
            blob = self.service.get_blob_to_text(containerName, blobName)
            returnContent = blob.content
        return returnContent

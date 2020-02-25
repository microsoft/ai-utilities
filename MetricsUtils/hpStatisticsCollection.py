"""
AI-Utilities - MetricUtils/blobStorage.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Enumeration used in teh statisticsCollector class to track individual tasks. These enums can be used by
both the producer (IPYNB path) and consumer (E2E path).

The statisticsCollector requires the enum for tracking calls
    startTask()
    endTask()
    addEntry()
    getEntry()
"""

import json
from datetime import datetime
from enum import Enum

from MetricsUtils.blobStorage import BlobStorageAccount
from MetricsUtils.storageutils import storageConnection

__version__ = "0.1"


class CollectionEntry(Enum):
    AKS_CLUSTER_CREATION = "akscreate"
    AML_COMPUTE_CREATION = "amlcompute"
    AML_WORKSPACE_CREATION = "amlworkspace"
    AKS_REALTIME_ENDPOINT = "aksendpoint"
    AKS_REALTIME_KEY = "akskey"


# Class used for keeping track of tasks during the execution of a path. Data can be archived and retrieved to/from
# Azure Storage.
#
# Tasks can be started and completed with an internal timer keeping track of the MS it takes to run
#     startTask()
#     endTask()
#
# Entries can also be added with any other data point the user would require
#     addEntry()
#
# Regardless of how an entry was put in the collection, it can be retrieved
#     getEntry()
#
# Working with the collection itself
#     getCollection() -> Retrieves the internal collection as JSON
#     uploadContent() -? Uploads to the provided storage account in pre-defined container/blob
#     hydrateFromStorage() -> Resets the internal collection to the data found in the provided storage account from
#     the pre-defined container/blob.


class statisticsCollector:
    """ Statistics Collector """
    __metrics__ = {}
    __runningTasks__ = {}
    __statscontainer__ = "pathmetrics"
    __statsblob__ = "statistics.json"

    # No need to create an instance, all methods are static.

    def __init__(self, pathName):
        self.pathName = pathName

    @staticmethod
    def startTask(collectionEntry):
        """
        Starts a task using one of the enumerators and records a start time of the task. If using this,
        the entry is not put into the __metrics__ connection until endTask() is called.

        :param collectionEntry: an instance of a CollectionEntry enum
        """
        statisticsCollector.__runningTasks__[collectionEntry.value] = datetime.utcnow()

    @staticmethod
    def endTask(collection_entry):
        """
        Ends a task using one of the enumerators. If the start time was previously recorded using
        startTask() an entry for the specific enumeration is added to the __metrics__ collection that
        will be used to upload data to Azure Storage.

        :param collection_entry: an instance of a CollectionEntry enum
        """
        if collection_entry.value in statisticsCollector.__runningTasks__.keys():
            timeDiff = datetime.utcnow() - statisticsCollector.__runningTasks__[collection_entry.value]
            msDelta = timeDiff.total_seconds() * 1000
            statisticsCollector.__metrics__[collection_entry.value] = msDelta

    '''

    '''

    @staticmethod
    def addEntry(collectionEntry, dataPoint):
        """
        Single call to add an entry to the __metrics__ collection. This would be used when you want to run
        the timers in the external code directly.

        This is used to set manual task times or any other valid data point.


        :param collectionEntry: an instance of a CollectionEntry enum
        :param dataPoint: Any valid python data type (string, int, etc)
        """
        statisticsCollector.__metrics__[collectionEntry.value] = dataPoint

    '''
        Retrieve an entry in the internal collection. 

        Parameters:
            collectionEntry - an instance of a CollectionEntry enum

        Returns:
            The data in the collection or None if the entry is not present.        
    '''

    @staticmethod
    def getEntry(collectionEntry):
        returnDataPoint = None
        if collectionEntry.value in statisticsCollector.__metrics__.keys():
            returnDataPoint = statisticsCollector.__metrics__[collectionEntry.value]
        return returnDataPoint

    '''
        Returns the __metrics__ collection as a JSON string.

        Parameters:
            None

        Returns:
            String representation of the collection in JSON
    '''

    @staticmethod
    def getCollection():
        return json.dumps(statisticsCollector.__metrics__)

    '''
        Uploads the JSON string representation of the __metrics__ collection to the specified
        storage account. 

        Parameters:
            connectionString - A complete connection string to an Azure Storage account

        Returns:
            Nothing
    '''

    @staticmethod
    def uploadContent(connectionString):
        connectionObject = storageConnection(connectionString)
        storageAccount = BlobStorageAccount(connectionObject)
        containers = storageAccount.getContainers()
        if statisticsCollector.__statscontainer__ not in containers:
            storageAccount.create_container(statisticsCollector.__statscontainer__)
        storageAccount.uploadBlob(statisticsCollector.__statscontainer__, statisticsCollector.__statsblob__,
                                  statisticsCollector.getCollection())

    '''
        Download the content from blob storage as a string representation of the JSON. This can be used for collecting
        and pushing downstream to whomever is interested. This call does not affect the internal collection.

        Parameters:
            connectionString - A complete connection string to an Azure Storage account

        Returns:
            The uploaded collection that was pushed to storage or None if not present.
        
    '''

    @staticmethod
    def retrieveContent(connectionString):
        returnContent = None
        connectionObject = storageConnection(connectionString)
        storageAccount = BlobStorageAccount(connectionObject)
        containers = storageAccount.getContainers()
        if statisticsCollector.__statscontainer__ in containers:
            returnContent = storageAccount.downloadBlob(statisticsCollector.__statscontainer__,
                                                        statisticsCollector.__statsblob__)
        return returnContent

    '''
        Retrieves the content in storage and hydrates the __metrics__ dictionary, dropping any existing information. 

        Useful between IPYNB runs/stages in DevOps.

        Parameters:
            connectionString - A complete connection string to an Azure Storage account

        Returns:
            Nothing
    '''

    @staticmethod
    def hydrateFromStorage(connectionString):
        returnContent = statisticsCollector.retrieveContent(connectionString)
        if returnContent is not None:
            statisticsCollector.__metrics__ = json.loads(returnContent)
        else:
            print("There was no data in storage")

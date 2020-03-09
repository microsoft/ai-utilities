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

from azure_utils.logger.blob_storage import BlobStorageAccount
from azure_utils.logger.storageutils import StorageConnection


class CollectionEntry(Enum):
    """ Deploy Steps Enums"""

    AKS_CLUSTER_CREATION = "akscreate"
    AML_COMPUTE_CREATION = "amlcompute"
    AML_WORKSPACE_CREATION = "amlworkspace"


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


class StatisticsCollector:
    """ Statistics Collector """

    __metrics__ = {}
    __running_tasks__ = {}
    __statscontainer__ = "pathmetrics"
    __statsblob__ = "statistics.json"

    # No need to create an instance, all methods are static.

    def __init__(self, path_name: str):
        self.path_name = path_name

    @staticmethod
    def start_task(collection_entry):
        """
        Starts a task using one of the enumerators and records a start time of the task. If using this,
        the entry is not put into the __metrics__ connection until endTask() is called.

        :param collection_entry: an instance of a CollectionEntry enum
        """
        StatisticsCollector.__running_tasks__[
            collection_entry.value
        ] = datetime.utcnow()

    @staticmethod
    def end_task(collection_entry):
        """
        Ends a task using one of the enumerators. If the start time was previously recorded using
        startTask() an entry for the specific enumeration is added to the __metrics__ collection that
        will be used to upload data to Azure Storage.

        :param collection_entry: an instance of a CollectionEntry enum
        """
        if collection_entry.value in StatisticsCollector.__running_tasks__.keys():
            time_diff = (
                datetime.utcnow()
                - StatisticsCollector.__running_tasks__[collection_entry.value]
            )
            ms_delta = time_diff.total_seconds() * 1000
            StatisticsCollector.__metrics__[collection_entry.value] = ms_delta

    @staticmethod
    def add_entry(collection_entry, data_point):
        """
        Single call to add an entry to the __metrics__ collection. This would be used when you want to run
        the timers in the external code directly.

        This is used to set manual task times or any other valid data point.


        :param collection_entry: an instance of a CollectionEntry enum
        :param data_point: Any valid python data type (string, int, etc)
        """
        StatisticsCollector.__metrics__[collection_entry.value] = data_point

    """
        Retrieve an entry in the internal collection. 

        Parameters:
            collectionEntry - an instance of a CollectionEntry enum

        Returns:
            The data in the collection or None if the entry is not present.        
    """

    @staticmethod
    def get_entry(collection_entry):
        """

        :param collection_entry:
        :return:
        """
        return_data_point = None
        if collection_entry.value in StatisticsCollector.__metrics__.keys():
            return_data_point = StatisticsCollector.__metrics__[collection_entry.value]
        return return_data_point

    """
        Returns the __metrics__ collection as a JSON string.

        Parameters:
            None

        Returns:
            String representation of the collection in JSON
    """

    @staticmethod
    def get_collection():
        """

        :return:
        """
        return json.dumps(StatisticsCollector.__metrics__)

    """
        Uploads the JSON string representation of the __metrics__ collection to the specified
        storage account. 

        Parameters:
            connectionString - A complete connection string to an Azure Storage account

        Returns:
            Nothing
    """

    @staticmethod
    def upload_content(connection_string):
        """

        :param connection_string:
        """
        containers, storage_account = StatisticsCollector._get_containers(
            connection_string
        )
        if StatisticsCollector.__statscontainer__ not in containers:
            storage_account.create_container(StatisticsCollector.__statscontainer__)
        storage_account.upload_blob(
            StatisticsCollector.__statscontainer__,
            StatisticsCollector.__statsblob__,
            StatisticsCollector.get_collection(),
        )

    """
        Download the content from blob storage as a string representation of the JSON. This can be used for collecting
        and pushing downstream to whomever is interested. This call does not affect the internal collection.

        Parameters:
            connectionString - A complete connection string to an Azure Storage account

        Returns:
            The uploaded collection that was pushed to storage or None if not present.
        
    """

    @staticmethod
    def retrieve_content(connection_string):
        """

        :param connection_string:
        :return:
        """
        return_content = None
        containers, storage_account = StatisticsCollector._get_containers(
            connection_string
        )
        if StatisticsCollector.__statscontainer__ in containers:
            return_content = storage_account.download_blob(
                StatisticsCollector.__statscontainer__,
                StatisticsCollector.__statsblob__,
            )
        return return_content

    @staticmethod
    def _get_containers(connection_string):
        connection_object = StorageConnection(connection_string)
        storage_account = BlobStorageAccount(connection_object)
        # noinspection PyUnresolvedReferences
        containers = storage_account.getContainers()
        return containers, storage_account

    """
        Retrieves the content in storage and hydrates the __metrics__ dictionary, dropping any existing information. 

        Useful between IPYNB runs/stages in DevOps.

        Parameters:
            connectionString - A complete connection string to an Azure Storage account

        Returns:
            Nothing
    """

    @staticmethod
    def hydrate_from_storage(connection_string):
        """

        :param connection_string:
        """
        return_content = StatisticsCollector.retrieve_content(connection_string)
        if return_content is not None:
            StatisticsCollector.__metrics__ = json.loads(return_content)
        else:
            print("There was no data in storage")

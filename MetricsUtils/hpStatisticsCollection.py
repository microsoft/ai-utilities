'''
	Copyright  Microsoft Corporation ("Microsoft").
	
	Microsoft grants you the right to use this software in accordance with your subscription agreement, if any, to use software 
	provided for use with Microsoft Azure ("Subscription Agreement").  All software is licensed, not sold.  
	
	If you do not have a Subscription Agreement, or at your option if you so choose, Microsoft grants you a nonexclusive, perpetual, 
	royalty-free right to use and modify this software solely for your internal business purposes in connection with Microsoft Azure 
	and other Microsoft products, including but not limited to, Microsoft R Open, Microsoft R Server, and Microsoft SQL Server.  
	
	Unless otherwise stated in your Subscription Agreement, the following applies.  THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT 
	WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL MICROSOFT OR ITS LICENSORS BE LIABLE 
	FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED 
	TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
	HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THE SAMPLE CODE, EVEN IF ADVISED OF THE
	POSSIBILITY OF SUCH DAMAGE.
'''

import json
import time
from enum import Enum
from datetime import datetime, timedelta
from MetricsUtils.storageutils import storageConnection
from MetricsUtils.blobStorage import blobStorageAccount

__version__ = "0.1"


'''
    Enumeratoin used in teh statisticsCollector class to track individual tasks. These enums can be used by 
    both the producer (IPYNB path) and consumer (E2E path). 

    The statisticsCollector requires the enum for tracking calls
        startTask()
        endTask()
        addEntry()
        getEntry()

'''
class CollectionEntry(Enum):
    AKS_CLUSTER_CREATION = "akscreate"
    AML_COMPUTE_CREATION = "amlcompute"
    AML_WORKSPACE_CREATION = "amlworkspace"
    AKS_REALTIME_ENDPOINT = "aksendpoint"
    AKS_REALTIME_KEY = "akskey"


'''
    Class used for keeping track of tasks during the execution of a path. Data can be archived and retrieved to/from Azure Storage.

    Tasks can be started and completed with an internal timer keeping track of the MS it takes to run
        startTask()
        endTask()

    Entries can also be added with any other data point the user would require
        addEntry()

    Regardless of how an entry was put in the collection, it can be retrieved
        getEntry()

    Working with the collection itself
        getCollection() -> Retrieves the internal collection as JSON
        uploadContent() -? Uploads to the provided storage account in pre-defined container/blob
        hydrateFromStorage() -> Resets the internal collection to the data found in the provided storage account from the pre-defined container/blob.
    
'''
class statisticsCollector :
    __metrics__ = {}
    __runningTasks__ = {}
    __statscontainer__ = "pathmetrics"
    __statsblob__ = "statistics.json"

    '''
        No need to create an instance, all methods are static.
    '''
    def __init__(self, pathName):
        self.pathName = pathName

    '''
        Starts a task using one of the enumerators and records a start time of the task. If using this,
        the entry is not put into the __metrics__ connection until endTask() is called.

        Parameters:
            collectionEntry - an instance of a CollectionEntry enum

        Returns:
            Nothing
    '''
    @staticmethod
    def startTask(collectionEntry):
        statisticsCollector.__runningTasks__[collectionEntry.value] = datetime.utcnow()

    '''
        Ends a task using one of the enumerators. If the start time was previously recorded using
        startTask() an entry for the specific enumeration is added to the __metrics__ collection that
        will be used to upload data to Azure Storage. 

        Parameters:
            collectionEntry - an instance of a CollectionEntry enum

        Returns:
            Nothing
    '''
    @staticmethod
    def endTask(collectionEntry):
        if collectionEntry.value in statisticsCollector.__runningTasks__.keys():
            timeDiff = datetime.utcnow() - statisticsCollector.__runningTasks__[collectionEntry.value]
            msDelta = timeDiff.total_seconds() * 1000
            statisticsCollector.__metrics__[collectionEntry.value] = msDelta   

    '''
        Single call to add an entry to the __metrics__ collection. This would be used when you want to run
        the timers in the external code directly. 

        This is used to set manual task times or any other valid data point. 

        Parameters:
            collectionEntry - an instance of a CollectionEntry enum
            dataPoint - Any valid python data type (string, int, etc)

        Returns:
            Nothing
    '''
    @staticmethod
    def addEntry(collectionEntry, dataPoint):
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
        storageAccount = blobStorageAccount(connectionObject)
        containers = storageAccount.getContainers()
        if statisticsCollector.__statscontainer__ not in containers:
            storageAccount.createContainer(statisticsCollector.__statscontainer__)
        storageAccount.uploadBlob(statisticsCollector.__statscontainer__,statisticsCollector.__statsblob__, statisticsCollector.getCollection())
        
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
        storageAccount = blobStorageAccount(connectionObject)
        containers = storageAccount.getContainers()
        if statisticsCollector.__statscontainer__ in containers:
            returnContent = storageAccount.downloadBlob(statisticsCollector.__statscontainer__,statisticsCollector.__statsblob__)
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
            statisticsCollector.__metrics__  = json.loads(returnContent)
        else:
            print("There was no data in storage")


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
__version__ = "0.1"

'''
    Class that performs work against an Azure Storage account with containers and blobs.

    To use, this library must be installed:

    pip install azure-storage-blob
'''

from datetime import datetime, timedelta
from azure.storage.blob import BlockBlobService, PublicAccess, BlobPermissions
from MetricsUtils.storageutils import storageConnection

class blobStorageAccount :
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
            returnToken = self.service.generate_blob_shared_access_signature(containerName, blobName, BlobPermissions.READ, datetime.utcnow() + timedelta(hours=1))

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
    def getBlobs(self,containerName):
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
            self.service.create_blob_from_text(containerName,blobName,fileContent)

    '''
        Download the blob as a string.
    '''
    def downloadBlob(self, containerName, blobName):
        returnContent = None
        if self.connection and self.service:
            blob = self.service.get_blob_to_text(containerName,blobName)
            returnContent = blob.content
        return returnContent

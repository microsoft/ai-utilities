
# ImportError: You need to install 'azure-cli-core' to load CLI active Cloud
from azure.mgmt.storage import StorageManagementClient
from azure.common.client_factory import get_client_from_cli_profile

__version__ = "0.1"


'''
    REQUIREMENT : pip install azure-cli-core

    Class that parses out a true connection string from Azure Storage account in the form:

    DefaultEndpointsProtocol=https;AccountName=ACCT_NAME;AccountKey=ACCT_KEY;EndpointSuffix=core.windows.net

    Ends up with 4 attributes : 
        DefaultEndpointsProtocol
        AccountName
        AccountKey
        EndpointSuffix
'''
class storageConnection:
    '''
        Constructor taking the connection string to parse.

        EX:
        DefaultEndpointsProtocol=https;AccountName=STGACCT_NAME;AccountKey=STGACCT_KEY;EndpointSuffix=core.windows.net

    '''
    def __init__(self, connectionString):
        parsedConnectionString = self._parseConnectionString(connectionString)
        for key, value in parsedConnectionString.items():
            self.__setattr__(key, value)


    '''
        Expects the full connection string from the Azure site and spits it into four components.

        EX:
        DefaultEndpointsProtocol=https;AccountName=STGACCT_NAME;AccountKey=STGACCT_KEY;EndpointSuffix=core.windows.net
    '''
    def _parseConnectionString(self,connectionString):
        returnValue = {}
        if connectionString:
                segments = connectionString.split(';')
                for segment in segments:
                        splitIndex = segment.index('=')
                        secondPart = (len(segment) - splitIndex - 1) * -1
                        returnValue[segment[:splitIndex]] = segment[secondPart:]

        return returnValue        

    '''
        Method to return the full connection string to an Azure Storage account give the resource group name and storage account
        name. 

        Method expects that the environment has been logged into Azue and the subscription has been set to match the incoming
        resource group and storage account. 
    '''
    @staticmethod 
    def getConnectionStringWithAzCredentials(resourceGroupName, storageAccountName):
        connectionStringTemplate = "DefaultEndpointsProtocol=https;AccountName={};AccountKey={};EndpointSuffix=core.windows.net"
        returnValue = None

        client = get_client_from_cli_profile(StorageManagementClient)
        keys = client.storage_accounts.list_keys(resourceGroupName,storageAccountName)
        key_value = None
        for key in keys.keys:
            key_value = key.value
            break
        
        if key_value is not None:
            returnValue = connectionStringTemplate.format(storageAccountName, key_value)

        return returnValue


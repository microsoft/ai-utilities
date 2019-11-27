from azure.mgmt.keyvault import KeyVaultManagementClient
from azure.common.client_factory import get_client_from_cli_profile
from azure.keyvault import KeyVaultClient
# pip install azure-keyvault
__version__ = "0.1"


class keyVaultInstance:
    '''
        Constructor taking the connection string to parse.

        EX:
        DefaultEndpointsProtocol=https;AccountName=STGACCT_NAME;AccountKey=STGACCT_KEY;EndpointSuffix=core.windows.net

    '''
    def __init__(self):
        self.__setattr__("Dan", "test")

    def getClient(self):
        self._kvmgmtClient =  get_client_from_cli_profile(KeyVaultManagementClient)
        return self._kvmgmtClient

    def getVaultNames(self):
        vaultName = []
        if self._kvmgmtClient is not None:
            for vlt in self._kvmgmtClient.vaults.list():
                vaultName.append(vlt.name)
        
        return vaultName

    def getKeyVltClient(self):
        self._vaultClient = get_client_from_cli_profile(KeyVaultClient) 
        return self._vaultClient

    def getVaultSecrets(self, vaultName):
        # https://thevault.vault.azure.net/
        returnSecrets = []
        vaultAddress = "https://{}.vault.azure.net/".format(vaultName)
        if self._vaultClient is not None:
            for sc in self._vaultClient.get_secrets(vaultAddress):
                scname = sc.id.split('/')[-1]
                scbundle = self._vaultClient.get_secret(vaultAddress, scname , "")
                scversion = scbundle.id.split('/')[-1]
                scvalue = scbundle.value    
                returnSecrets.append((scname, scversion,scvalue))

        return returnSecrets

    def setVaultSecret(self, vaultName, secretName, secretValue):
        # https://thevault.vault.azure.net/
        returnSecrets = []
        vaultAddress = "https://{}.vault.azure.net/".format(vaultName)
        if self._vaultClient is not None:
            self._vaultClient.set_secret(vaultAddress, secretName, secretValue)

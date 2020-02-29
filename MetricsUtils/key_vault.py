from azure.common.client_factory import get_client_from_cli_profile
from azure.keyvault import KeyVaultClient
from azure.mgmt.keyvault import KeyVaultManagementClient

# pip install azure-keyvault
__version__ = "0.1"


class KeyVaultInstance:
    """
        Constructor taking the connection string to parse.

        EX:
        DefaultEndpointsProtocol=https;AccountName=STGACCT_NAME;AccountKey=STGACCT_KEY;EndpointSuffix=core.windows.net

    """

    def __init__(self):
        self._vault_client = get_client_from_cli_profile(KeyVaultClient)
        self._kvmgmt_client = get_client_from_cli_profile(KeyVaultManagementClient)
        self.__setattr__("Dan", "test")

    def get_client(self):
        return self._kvmgmt_client

    def get_vault_names(self):
        vault_name = []
        if self._kvmgmt_client is not None:
            for vlt in self._kvmgmt_client.vaults.list():
                vault_name.append(vlt.name)

        return vault_name

    def get_key_vlt_client(self):
        return self._vault_client

    def get_vault_secrets(self, vault_name):
        # https://thevault.vault.azure.net/
        return_secrets = []
        vault_address = "https://{}.vault.azure.net/".format(vault_name)
        if self._vault_client is not None:
            for sc in self._vault_client.get_secrets(vault_address):
                scname = sc.id.split('/')[-1]
                scbundle = self._vault_client.get_secret(vault_address, scname, "")
                scversion = scbundle.id.split('/')[-1]
                scvalue = scbundle.value
                return_secrets.append((scname, scversion, scvalue))

        return return_secrets

    def set_vault_secret(self, vault_name, secret_name, secret_value):
        # https://thevault.vault.azure.net/
        vault_address = "https://{}.vault.azure.net/".format(vault_name)
        if self._vault_client is not None:
            self._vault_client.set_secret(vault_address, secret_name, secret_value)

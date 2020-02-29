import time

from MetricsUtils.hpStatisticsCollection import statisticsCollector, CollectionEntry
from MetricsUtils.keyVault import KeyVaultInstance

from MetricsUtils.storageutils import StorageConnection

__version__ = "0.1"
DATA_IN_STORAGE_ = "Current data in storage ->"

kvInst = KeyVaultInstance()

cl = kvInst.get_client()
names = kvInst.get_vault_names()
kvc = kvInst.get_key_vlt_client()

sct = kvInst.get_vault_secrets("dangtestvault")
print(sct)
sct = kvInst.set_vault_secret("dangtestvault", "secret2", "asecretvalue")
sct = kvInst.get_vault_secrets("dangtestvault")
print(sct)

exit(1)

# You will need a connection string to access the storage account. This can be done by either supplying the
# connection string
# in total, or, assuming your az cli has logged in to the appropriate subscription, you can generate one using the
# storage client.


storageConnString = None
storageResourceGroup = "dangtest"
storageAccountName = "hpstatstest"

if storageConnString is None:
    storageConnString = StorageConnection.get_connection_string_with_az_credentials(storageResourceGroup,
                                                                                    storageAccountName)

'''
    The stattisticsCollector is used in any python path you want. It's used for timing specific tasks and saving 
    the results to blob storage, then pulling that data from blob storage. 

    Timings can be collected in two ways
    1. By starting and stopping a task
    2. By simply putting in the time it took to run. 

    Now, you can use a single instance to collect data across a single execution, or you can use it to get data and 
    append to it
    between execution runs (think IPYNB seperate executions)
'''

'''
    Tests with putting in time indirectly
'''
statisticsCollector.start_task(CollectionEntry.AML_COMPUTE_CREATION)
time.sleep(1.5)
statisticsCollector.end_task(CollectionEntry.AML_COMPUTE_CREATION)

# Upload the content to storage
statisticsCollector.upload_content(storageConnString)

# Retrieve the content from storage
content = statisticsCollector.retrieve_content(storageConnString)
print(DATA_IN_STORAGE_)
print(content)
print("")

'''
    Tests with putting in time directly
'''

statisticsCollector.add_entry(CollectionEntry.AKS_CLUSTER_CREATION, 200)
statisticsCollector.add_entry(CollectionEntry.AML_COMPUTE_CREATION, 200)
statisticsCollector.add_entry(CollectionEntry.AML_WORKSPACE_CREATION, 200)

# Upload the content to storage
statisticsCollector.upload_content(storageConnString)

# Retrieve the content from storage
content = statisticsCollector.retrieve_content(storageConnString)
print(DATA_IN_STORAGE_)
print(content)
print("")

'''
    Work with the data in storage and append to it..

    First change a bunch of data so we know it's not cached....
'''
statisticsCollector.add_entry(CollectionEntry.AKS_CLUSTER_CREATION, 0)
statisticsCollector.add_entry(CollectionEntry.AML_COMPUTE_CREATION, 0)

statisticsCollector.hydrate_from_storage(storageConnString)

# Now change a 200 to 300
statisticsCollector.add_entry(CollectionEntry.AML_WORKSPACE_CREATION, 300)

# Upload the content to storage
statisticsCollector.upload_content(storageConnString)

# Retrieve the content from storage
content = statisticsCollector.retrieve_content(storageConnString)
print(DATA_IN_STORAGE_)
print(content)
print("")

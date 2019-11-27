import time
from MetricsUtils.hpStatisticsCollection import statisticsCollector, CollectionEntry
from MetricsUtils.storageutils import storageConnection
from MetricsUtils.keyVault import keyVaultInstance

__version__ = "0.1"

kvInst = keyVaultInstance()

cl = kvInst.getClient()
names = kvInst.getVaultNames()
kvc = kvInst.getKeyVltClient()

sct = kvInst.getVaultSecrets("dangtestvault")
print(sct)
sct = kvInst.setVaultSecret("dangtestvault", "secret2", "asecretvalue")
sct = kvInst.getVaultSecrets("dangtestvault")
print(sct)

'''
for vlt in cl.vaults.list():
    print(vlt.id)
    print(vlt.name)
'''
exit(1)

'''
    You will need a connection string to access the storage account. This can be done by either supplying the connection string 
    in total, or, assuming your az cli has logged in to the appropriate subscription, you can generate one using the storage client.
'''
#storageConnString = "DefaultEndpointsProtocol=https;AccountName=ACCOUNT_NAME;AccountKey=ACCOUNT_KEY;EndpointSuffix=core.windows.net"
storageConnString = None
storageResourceGroup = "dangtest"
storageAccountName = "hpstatstest"

if storageConnString is None:
    storageConnString = storageConnection.getConnectionStringWithAzCredentials(storageResourceGroup, storageAccountName)

'''
    The stattisticsCollector is used in any python path you want. It's used for timing specific tasks and saving 
    the results to blob storage, then pulling that data from blob storage. 

    Timings can be collected in two ways
    1. By starting and stopping a task
    2. By simply putting in the time it took to run. 

    Now, you can use a single instance to collect data across a single execution, or you can use it to get data and append to it
    between execution runs (think IPYNB seperate executions)
'''


'''
    Tests with putting in time indirectly
'''
statisticsCollector.startTask(CollectionEntry.AML_COMPUTE_CREATION)
time.sleep(1.5)
statisticsCollector.endTask(CollectionEntry.AML_COMPUTE_CREATION)

# Upload the content to storage
statisticsCollector.uploadContent(storageConnString)

# Retrieve the content from storage
content = statisticsCollector.retrieveContent(storageConnString)
print("Current data in storage ->")
print(content)
print("")

'''
    Tests with putting in time directly
'''

statisticsCollector.addEntry(CollectionEntry.AKS_CLUSTER_CREATION, 200)
statisticsCollector.addEntry(CollectionEntry.AML_COMPUTE_CREATION, 200)
statisticsCollector.addEntry(CollectionEntry.AML_WORKSPACE_CREATION, 200)

# Upload the content to storage
statisticsCollector.uploadContent(storageConnString)

# Retrieve the content from storage
content = statisticsCollector.retrieveContent(storageConnString)
print("Current data in storage ->")
print(content)
print("")

'''
    Work with the data in storage and append to it..

    First change a bunch of data so we know it's not cached....
'''
statisticsCollector.addEntry(CollectionEntry.AKS_CLUSTER_CREATION, 0)
statisticsCollector.addEntry(CollectionEntry.AML_COMPUTE_CREATION, 0)

statisticsCollector.hydrateFromStorage(storageConnString)

# Now change a 200 to 300
statisticsCollector.addEntry(CollectionEntry.AML_WORKSPACE_CREATION, 300)


# Upload the content to storage
statisticsCollector.uploadContent(storageConnString)

# Retrieve the content from storage
content = statisticsCollector.retrieveContent(storageConnString)
print("Current data in storage ->")
print(content)
print("")

# Data Tracker
<sub>Dan Grecoe - A Microsoft Employee </sub>

When running Python projects through Azure Dev Ops (https://dev.azure.com) there is a need to collect certain statistics
such as deployment time, or to pass out information related to a deployment as the agent the build runs on will be torn
down once the build os complete.

Of course, there are several option for doing so and this repository contains one option.

The code in this repository enables saving these data to an Azure Storage account for consumption at a later time.

Descriptions of the class and how it performs can be found in the MetricsUtils/hpStatisticsCollection.py file. 

An example on how to use the code for various tasks can be found in the statsCollectionTest.py file. 

## Pre-requisites
To use this example, you must pip install the following into your environment:
    - azure-cli-core
    - azure-storage-blob

These should be installed with the azml libraries, but if they don't work that is why.

## Use in a notebook with AZML
First you need to include the following 

```
    from MetricsUtils.hpStatisticsCollection import statisticsCollector, CollectionEntry
    from MetricsUtils.storageutils import storageConnection
```

This gives you access to the code. This assumes that you have installed either as a submodule or manually, the files in
a folder called MetricsUtils in the same directory as the notebooks themselves.

### First notebook
In the first notebook, you can certainly make use of the tracker to collect stats before the workspace is created, for
example: 

```
statisticsCollector.startTask(CollectionEntry.AML_WORKSPACE_CREATION)

ws = Workspace.create(
    name=workspace_name,
    subscription_id=subscription_id,
    resource_group=resource_group,
    location=workspace_region,
    create_resource_group=True,
    auth=get_auth(env_path),
    exist_ok=True,
)

statisticsCollector.endTask(CollectionEntry.AML_WORKSPACE_CREATION)
```

In fact, you are going to need to create this workspace to get the storage account name. So, in that first notebook, you
 will likely want to save off the storage connection string into the environment or .env file. 

The storage account name can be found with this code:
```
stgAcctName = ws.get_details()['storageAccount'].split('/')[-1]
```

Once you have the storage account name, you save the statistics to storage using the following at or near the bottom of
your notebook. If you believe there may be failures along the way, you can perform the upload multiple times, it will
just overwrite what is there.

Also note that this assumes the user is logged in to the same subscription as the storage account.
```
storageConnString = storageConnection.getConnectionStringWithAzCredentials(resource_group, stgAcct)
statisticsCollector.uploadContent(storageConnString)
```

### Follow on notebooks
The difference in a follow up notebook is that settings have likely already been saved. Since we have the storage
account name now in the environment, we just need to pull the information from storage into the tracking class such as:

```
storageConnString = storageConnection.getConnectionStringWithAzCredentials(resource_group, stgAcct)
statisticsCollector.hydrateFromStorage(storageConnString)
```

Then continue to use the object as you did in the first notebook being sure to call teh uploadContent() method to save
whatever changes you want to storage.

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

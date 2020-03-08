# Configuration
Obtaining user information is critical to any project you will produce. At a minimum it is required to get an Azure Subscription but often it is important to collect many settings for a project to be successful. 

In the past we have used a combination of dotenv and cookiecutter. While these can be extremely useful, this configuration code provides a user interface for the user to provide information in a yml file. 

The structure of the yml file is as follows:

```
project_name: AI Default Project
settings:
- subscription_id:
  - description: Azure Subscription Id 
  - value: <>
- resource_group:
  - description: Azure Resource Group Name 
  - value: <>
[etc, continue adding settings as needed ]
```

### Scripts
|Name|Description|
|------|------|
|configuration.py|Contains a class called ProjectConfiguration. This class manages reading/writing the configuration settings file.|
|configurationui.py|Contains a class called SettingsUpdate. This class reads any valid configuration file as defined by the yml structure. It dynamically builds a tkinter UI displaying the description of each setting and the ability for the user to input new values.|
|config_tests.py|Unit tests for ProjectConfiguration.|
|notebook_config.py|Provides a function to add into IPython Notebooks to simply add in UI driven configuration settings collection from the end user.|
|exampleconfiguration.ipynb|Example IPython Notebook that utilizes the configuration settings objects.|
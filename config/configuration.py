import os
import yaml


class ProjectConfiguration:
    '''
        Configuration file is formed as:

        project_name: AMLS Test
        settings :
          - subscription_id:
            - description : Azure Subscription Id
            - value: <>    
          - resource_group:
            - description : Azure Resource Group Name
            - value: <>    
        etc....
    '''
    project_key = "project_name"
    settings_key = "settings"
    setting_value = 'value'
    setting_description = 'description'

    def __init__(self, configuration_file):
        '''
            Sets up the configuration file. If it does not exist, it is created 
            with a default name and no settings.
        '''
        self.configuration_file = configuration_file
        self.configuration = None

        if os.path.isfile(self.configuration_file) == False:
            self.configuration = {ProjectConfiguration.project_key : "Default Settings", ProjectConfiguration.settings_key : None}
            self.save_configuration()

        self._load_configuration()

    def _validateConfiguration(self, key_name):
        '''
            Ensure configuration has been loaded with load_configuration, and 
            that the given top level key exists.

            There are only two keys we care about:
                ProjectConfiguration.project_key
                ProjectConfiguration.settings_key
        '''
        if self.configuration == None:
            raise Exception("Load configuration file first")

        if key_name not in self.configuration.keys():
            raise Exception("Invalid configuration file")

    def _load_configuration(self):
        '''
            Load the configuration file from disk, there is no security around this. Although
            it will be called from the constructor, which will create a default file for the user.
        '''
        with open(self.configuration_file, 'r') as ymlfile:
            self.configuration = yaml.load(ymlfile, Loader=yaml.BaseLoader)

        assert self.configuration

    def project_name(self):
        '''
            Get the configured project name
        '''
        self._validateConfiguration(ProjectConfiguration.project_key)
        return self.configuration[ProjectConfiguration.project_key]

    def set_project_name(self, project_name):
        '''
            Set the project name
        '''
        self._validateConfiguration(ProjectConfiguration.project_key)
        self.configuration[ProjectConfiguration.project_key] = project_name

    def get_settings(self):
        '''
            Get all of the settings (UI Configuration)
        '''
        self._validateConfiguration(ProjectConfiguration.settings_key)
        return self.configuration[ProjectConfiguration.settings_key]

    def add_setting(self, setting_name, description, value):
        '''
            Add a setting to the configuration. A setting consists of:
                {
                    name: [
                        {ProjectConfiguration.setting_description : description},
                        {ProjectConfiguration.setting_value : value}
                    ]
                }
        '''
        self._validateConfiguration(ProjectConfiguration.settings_key)

        if isinstance(self.configuration[ProjectConfiguration.settings_key], list) == False:
            self.configuration[ProjectConfiguration.settings_key] = []
        
        new_setting = {setting_name : []}
        new_setting[setting_name].append({ProjectConfiguration.setting_description : description})
        new_setting[setting_name].append({ProjectConfiguration.setting_value : value})
        self.configuration[ProjectConfiguration.settings_key].append(new_setting)

    def get_value(self, setting_name):
        '''
            Get the value of a specific setting. If the file has no settings or does not contain 
            this specific setting return None, otherwise return the value. 
        '''
        self._validateConfiguration(ProjectConfiguration.settings_key)

        return_value = None

        if isinstance(self.configuration[ProjectConfiguration.settings_key], list):
            setting = [x for x in self.configuration[ProjectConfiguration.settings_key] if setting_name in x.keys()]
            if len(setting) == 1:
                value = [x for x in setting[0][setting_name] if ProjectConfiguration.setting_value in x.keys()]
                if len(value) == 1:
                    return_value = value[0][ProjectConfiguration.setting_value]
        
        return return_value

    def set_value(self, setting_name, value):
        '''
            Set the value of a specific setting. However, if this is just created there is no setting to set 
            and the request is silently ignored. 
        '''
        self._validateConfiguration(ProjectConfiguration.settings_key)

        if isinstance(self.configuration[ProjectConfiguration.settings_key], list):
            setting = [x for x in self.configuration[ProjectConfiguration.settings_key] if setting_name in x.keys()]
            if len(setting) == 1:
                current_value = [x for x in setting[0][setting_name] if ProjectConfiguration.setting_value in x.keys()]
                if len(current_value) == 1:
                    current_value[0][ProjectConfiguration.setting_value] = value
                else:
                    value_setting = {ProjectConfiguration.setting_value : value} 
                    setting[0][setting_name].append(value_setting)

    def save_configuration(self):
        '''
            Save the configuration file
        '''
        with open(self.configuration_file, 'w') as ymlfile:
            yaml.dump(self.configuration, ymlfile)

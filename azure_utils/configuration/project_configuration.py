"""
- project_configuration.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import os
from typing import Optional, Dict

import yaml


class ProjectConfiguration:
    """
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
    """
    configuration: Dict[str, Optional[str]]
    project_key = "project_name"
    settings_key = "settings"
    setting_value = 'value'
    setting_description = 'description'

    def __init__(self, configuration_file: str):
        """
        Sets up the configuration file. If it does not exist, it is created
        with a default name and no settings.

        :param configuration_file: File path to configuration file
        """
        found, file_dir = find_file(configuration_file)
        self.configuration_file = file_dir + "/" + configuration_file
        self.configuration = {}

        if not found:
            self.configuration = {ProjectConfiguration.project_key: "Default Settings",
                                  ProjectConfiguration.settings_key: None}
            self.save_configuration()

        self._load_configuration()

    def _validate_configuration(self, key_name: str):
        """
        Ensure configuration has been loaded with load_configuration, and
        that the given top level key exists.

        There are only two keys we care about:
            ProjectConfiguration.project_key
            ProjectConfiguration.settings_key

        :param key_name: Configuration key who's existence will be checked.
        """
        if self.configuration is None:
            raise Exception("Load configuration file first")

        if key_name not in self.configuration.keys():
            raise Exception("Invalid configuration file")

    def _load_configuration(self):
        """
        Load the configuration file from disk, there is no security around this. Although
        it will be called from the constructor, which will create a default file for the user.
        """
        with open(self.configuration_file, 'r') as ymlfile:
            self.configuration = yaml.load(ymlfile, Loader=yaml.BaseLoader)

        assert self.configuration

    def project_name(self) -> str:
        """
        Get the configured project name

        :return: Configured Project Name
        """
        self._validate_configuration(ProjectConfiguration.project_key)
        return self.configuration[ProjectConfiguration.project_key]

    def set_project_name(self, project_name: str):
        """
        Set the project name

        :param project_name: Project Configuration Name
        """
        self._validate_configuration(ProjectConfiguration.project_key)
        self.configuration[ProjectConfiguration.project_key] = project_name

    def get_settings(self) -> Optional[str]:
        """
        Get all of the settings (UI Configuration)

        :return:  Return UI Configuration
        """
        self._validate_configuration(ProjectConfiguration.settings_key)
        return self.configuration[ProjectConfiguration.settings_key]

    def add_setting(self, setting_name: str, description: str, value: str):
        """
        Add a setting to the configuration. A setting consists of:
            {
                name: [
                    {ProjectConfiguration.setting_description : description},
                    {ProjectConfiguration.setting_value : value}
                ]
            }

        :param setting_name: Name of setting key
        :param description: Text describing the setting
        :param value: Value of setting to saving in configuration
        """
        self._validate_configuration(ProjectConfiguration.settings_key)

        if not isinstance(self.configuration[ProjectConfiguration.settings_key], list):
            self.configuration[ProjectConfiguration.settings_key] = []

        new_setting = {setting_name: []}
        new_setting[setting_name].append({ProjectConfiguration.setting_description: description})
        new_setting[setting_name].append({ProjectConfiguration.setting_value: value})
        self.configuration[ProjectConfiguration.settings_key].append(new_setting)

    def get_value(self, setting_name: str) -> str:
        """
        Get the value of a specific setting. If the file has no settings or does not contain
        this specific setting return None, otherwise return the value.

        :param setting_name: Key of setting to return
        :return: Value of requested setting_name
        """
        self._validate_configuration(ProjectConfiguration.settings_key)

        return_value = None

        if isinstance(self.configuration[ProjectConfiguration.settings_key], list):
            setting = [x for x in self.configuration[ProjectConfiguration.settings_key] if setting_name in x.keys()]
            if len(setting) == 1:
                value = [x for x in setting[0][setting_name] if ProjectConfiguration.setting_value in x.keys()]
                if len(value) == 1:
                    return_value = value[0][ProjectConfiguration.setting_value]

        return return_value

    def set_value(self, setting_name: str, value: str):
        """
        Set the value of a specific setting. However, if this is just created there is no setting to set
        and the request is silently ignored.

        :param setting_name: Key of setting to set
        :param value: Value of setting to set
        """
        self._validate_configuration(ProjectConfiguration.settings_key)

        if isinstance(self.configuration[ProjectConfiguration.settings_key], list):
            setting = [x for x in self.configuration[ProjectConfiguration.settings_key] if setting_name in x.keys()]
            if len(setting) == 1:
                current_value = [x for x in setting[0][setting_name] if ProjectConfiguration.setting_value in x.keys()]
                if len(current_value) == 1:
                    current_value[0][ProjectConfiguration.setting_value] = value
                else:
                    value_setting = {ProjectConfiguration.setting_value: value}
                    setting[0][setting_name].append(value_setting)

    def save_configuration(self):
        """ Save the configuration file """
        with open(self.configuration_file, 'w') as ymlfile:
            yaml.dump(self.configuration, ymlfile)


def transverse_up(file: str, search_depth: int = 5):
    """
    Check if file is in directory, and if not recursive call up to 5 times

    :param file: Configuration File Name
    :param search_depth: Number of directories to search up through
    """

    if search_depth == 0:
        return False
    if not os.path.isfile(file):
        os.chdir("../")
        transverse_up(file, search_depth=search_depth - 1)
    if os.path.isfile(file):
        return True
    return False


def find_file(file: str):
    """
    Transverse up directories to try and find configuration file

    :param file: Configuration File Name
    """
    curdir = os.path.abspath(os.curdir)
    found = transverse_up(file)
    if found:
        file_dir = os.path.abspath(os.curdir)
    else:
        file_dir = curdir
    os.chdir(curdir)
    return found, file_dir

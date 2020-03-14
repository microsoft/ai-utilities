"""
- project_configuration.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import os
from typing import Dict, Optional

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
    setting_value = "value"
    setting_description = "description"

    def __init__(self, configuration_file: str):
        """
        Sets up the configuration file. If it does not exist, it is created
        with a default name and no settings.

        :param configuration_file: File path to configuration file
        """
        self.configuration: dict = {}
        found, file_dir = find_file(configuration_file)
        self.configuration_file = file_dir + "/" + configuration_file
        if not found:
            self.set_project_name("project_name")
            self.add_setting("subscription_id", "Your Azure Subscription", "<>")
            self.add_setting("resource_group", "Azure Resource Group Name", "<>")
            self.add_setting("workspace_name", "Azure ML Workspace Name", "<>")
            self.add_setting("workspace_region", "Azure ML Workspace Region", "<>")
            self.add_setting("image_name", "Docker Container Image Name", "<>")
            self.add_setting("aks_service_name", "AKS Service Name", "<>")
            self.add_setting("aks_location", " AKS Azure Region", "<>")
            self.add_setting("aks_name", "AKS Cluster Name", "<>")
            self.add_setting("deep_image_name", "Docker Container Image Name", "<>")
            self.add_setting("deep_aks_service_name", "AKS Service Name", "<>")
            self.add_setting("deep_aks_name", "AKS Cluster Name", "<>")
            self.add_setting("deep_aks_location", "AKS Azure Region", "<>")

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

    def _load_configuration(self) -> None:
        """
        Load the configuration file from disk, there is no security around this. Although
        it will be called from the constructor, which will create a default file for the user.
        """
        with open(self.configuration_file) as ymlfile:
            self.configuration = yaml.safe_load(ymlfile)

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
        # self._validate_configuration(ProjectConfiguration.project_key)
        self.configuration[ProjectConfiguration.project_key] = project_name

    def has_value(self, setting_name: str) -> bool:
        """
        Get all of the settings (UI Configuration)

        :param setting_name: Key of setting to return
        :return:  Return UI Configuration
        """
        if self.get_value(setting_name):
            return True
        return False

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
        # self._validate_configuration(ProjectConfiguration.settings_key)

        if (
            ProjectConfiguration.settings_key not in self.configuration
            or not isinstance(
                self.configuration[ProjectConfiguration.settings_key], list
            )
        ):
            # noinspection PyTypeChecker
            self.configuration[ProjectConfiguration.settings_key] = []

        new_setting = {setting_name: []}
        new_setting[setting_name].append(
            {ProjectConfiguration.setting_description: description}
        )
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
            setting = self.get_settings_from_config(setting_name)
            if len(setting) == 1:
                value = self.get_value_from_config(setting, setting_name)
                if len(value) == 1:
                    return_value = value[0][ProjectConfiguration.setting_value]

        return return_value

    def get_settings_from_config(self, setting_name: str) -> list:
        """
        Get list of project settings from yaml configuration.

        :param setting_name: Name of settings key in YAML file
        :return: list of project settings
        """
        setting = [
            x
            for x in self.configuration[ProjectConfiguration.settings_key]
            if setting_name in x.keys()
        ]
        return setting

    def set_value(self, setting_name: str, value: str):
        """
        Set the value of a specific setting. However, if this is just created there is no setting to set
        and the request is silently ignored.

        :param setting_name: Key of setting to set
        :param value: Value of setting to set
        """
        self._validate_configuration(ProjectConfiguration.settings_key)

        if isinstance(self.configuration[ProjectConfiguration.settings_key], list):
            setting = self.get_settings_from_config(setting_name)
            if len(setting) == 1:
                current_value = self.get_value_from_config(setting, setting_name)
                if len(current_value) == 1:
                    current_value[0][ProjectConfiguration.setting_value] = value
                else:
                    value_setting = {ProjectConfiguration.setting_value: value}
                    # noinspection PyTypeChecker
                    setting[0][setting_name].append(value_setting)


    def append_value(self, setting_name: str, value: str):
        """
        Append the value of a specific setting. However, if this is just created there is no setting to set
        and the request is silently ignored.

        :param setting_name: Key of setting to set
        :param value: Value of setting to set
        """
        original_value = self.get_value(setting_name=setting_name)
        self.set_value(setting_name=setting_name, value=original_value + value)

    @staticmethod
    def get_value_from_config(setting: list, setting_name: str) -> list:
        """
        Get a setting from the list of settings

        :param setting: list of settings
        :param setting_name: name of setting to return
        :return: settings of given name
        """
        return [
            x
            for x in setting[0][setting_name]
            if ProjectConfiguration.setting_value in x.keys()
        ]

    def save_configuration(self) -> None:
        """ Save the configuration file """
        with open(self.configuration_file, "w") as ymlfile:
            yaml.dump(self.configuration, ymlfile)


def transverse_up(file: str, search_depth: int = 3):
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

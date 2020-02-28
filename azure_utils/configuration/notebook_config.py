"""
- notebook_config.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Import the needed functionality
- tkinter :
    Python GUI library
- configuration.ProjectConfiguration
    Configuration object that reads/writes to the configuration settings YAML file.
- configurationui.SettingsUpdate
    tkinter based UI that dynamically loads any appropriate configuration file
    and displays it to the user to alter the settings.


    If you wish to run this file locally, uncomment the section below and then run
    the Python script directly from this directory. This will utilize the
    project.yml file as the configuration file under test.

LOCAL_ONLY
import os
import sys
if __name__ == "__main__":
    current = os.getcwd()
    az_utils = os.path.split(current)
    while not az_utils[0].endswith("AI-Utilities"):
        az_utils = os.path.split(az_utils[0])

    if az_utils[0] not in sys.path:
        sys.path.append(az_utils[0])
"""

from tkinter import Tk

from azure_utils.configuration.configuration_ui import SettingsUpdate
from azure_utils.configuration.project_configuration import ProjectConfiguration

project_configuration_file = "project.yml"


def get_or_configure_settings(configuration_yaml: str = project_configuration_file):
    """
    Only configure the settings if the subscription ID has not been provided yet.
    This will help with automation in which the configuration file is provided.

    :param configuration_yaml: Location of configuration yaml
    """
    settings_object = get_settings(configuration_yaml)
    sub_id = settings_object.get_value('subscription_id')

    if sub_id == '<>':
        configure_settings(configuration_yaml)

    return get_settings(configuration_yaml)


def configure_settings(configuration_yaml: str = project_configuration_file):
    """
        Launch a tkinter UI to configure the project settings in the provided
        configuration_yaml file. If a file is not provided, the default ./project.yml
        file will be created for the caller.

        configuration_yaml -> Disk location of the configuration file to modify.

        ProjectConfiguration will open an existing YAML file or create a new one. It is
        suggested that your project simply create a simple configuration file containing
        all of you settings so that the user simply need to modify it with the UI.

        In this instance, we assume that the default configuration file is called project.yml.
        This will be used if the user passes nothing else in.

        :param configuration_yaml: Location of configuration yaml
    """
    project_configuration = ProjectConfiguration(configuration_yaml)

    # Finally, create a Tk window and pass that along with the configuration object
    # to the SettingsObject class for modification.

    window = Tk()
    app = SettingsUpdate(project_configuration, window)
    app.mainloop()


def get_settings(configuration_yaml: str = project_configuration_file) -> ProjectConfiguration:
    """
        Aquire the project settings from the provided configuration_yaml file.
        If a file is not provided, the default ./project.yml will be created and
        and empty set of settings will be returned to the user.

        configuration_yaml -> Disk location of the configuration file to modify.

        ProjectConfiguration will open an existing YAML file or create a new one. It is
        suggested that your project simply create a simple configuration file containing
        all of you settings so that the user simply need to modify it with the UI.

        In this instance, we assume that the default configuration file is called project.yml.
        This will be used if the user passes nothing else in.

        :param configuration_yaml: Project configuration yml
        :return: loaded ProjectConfiguration object
    """
    return ProjectConfiguration(configuration_yaml)


if __name__ == '__main__':
    configure_settings()

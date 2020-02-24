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
"""

from tkinter import *

from azure_utils.configuration.configuration_ui import SettingsUpdate
from azure_utils.configuration.project_configuration import ProjectConfiguration

project_configuration_file = "project.yml"


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

    '''
        Finally, create a Tk window and pass that along with the configuration object
        to the SettingsObject class for modification. 
    '''
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

'''
    Import the needed functionality
    - tkinter : 
        Python GUI library
    - configuration.ProjectConfiguration
        Configuration object that reads/writes to the configuration settings YAML file.
    - configurationui.SettingsUpdate
        tkinter based UI that dynamically loads any appropriate configuration file
        and displays it to the user to alter the settings.
'''
from tkinter import * 
from configuration import ProjectConfiguration
from configurationui import SettingsUpdate

project_configuration_file = "./project.yml"

def configure_settings(configuration_yaml = None):
    '''
        ProjectConfiguration will open an existing YAML file or create a new one. It is
        suggested that your project simply create a simple configuration file containing
        all of you settings so that the user simply need to modify it with the UI. 

        In this instance, we assume that the default configuration file is called project.yml.
        This will be used if the user passes nothing else in. 
    '''
    global project_configuration_file
    if configuration_yaml:
        project_configuration_file = configuration_yaml

    project_configuration = ProjectConfiguration(project_configuration_file)

    '''
        Finally, create a Tk window and pass that along with the configuration object
        to the SettingsObject class for modification. 
    '''
    window = Tk()
    app = SettingsUpdate(project_configuration, window)
    app.mainloop()

def get_settings(configuration_yaml = None):
    '''
        ProjectConfiguration will open an existing YAML file or create a new one. It is
        suggested that your project simply create a simple configuration file containing
        all of you settings so that the user simply need to modify it with the UI. 

        In this instance, we assume that the default configuration file is called project.yml.
        This will be used if the user passes nothing else in. 
    '''
    global project_configuration_file
    if configuration_yaml:
        project_configuration_file = configuration_yaml

    return ProjectConfiguration(project_configuration_file)

if __name__ == '__main__':
    configure_settings()
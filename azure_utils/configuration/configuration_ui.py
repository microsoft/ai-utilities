"""
- configuraitonui.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from tkinter import *

from azure_utils.configuration.project_configuration import ProjectConfiguration


class SettingsUpdate(Frame):
    """
    UI Wrapper for project configuration settings.

    Provide a configuraiton file as described in configuration.ProjectConfiguration.

    A UI is built using a grid where each row consists of:
        setting_description | Text control to show accept values

    Final row of the grid has a save and cancel button.

    Save updates the configuration file with any settings put on the UI.
    """

    def __init__(self, project_configuration, master):
        """
        Create new Configuration UI

        :param project_configuration: path to project configuration
        :param master: Main Widget
        """
        Frame.__init__(self, master=master)
        """
            self.configuration  = Instance of ProjectConfiguration and master
            self.master_win     = Instance of Tk application. 
            self.settings       = Will be a dictionary where
                                    key = Setting name
                                    value = Text control   
        """
        self.configuration = project_configuration
        self.master_win = master
        self.settings = {}

        '''
            Set up some window options
        '''
        self.master_win.title(self.configuration.project_name())
        self.master_win.resizable(width=TRUE, height=TRUE)
        self.master_win.configure(padx=10, pady=10)

        '''
            Populate the grid first with settings followed by the two buttons (cancel/save)
        '''
        current_row = 0
        for setting in self.configuration.get_settings():

            if not isinstance(setting, dict):
                print("Found setting does not match pattern...")
                continue

            # Only can be one key as they are sinletons with a list
            # of values
            if len(setting.keys()) == 1:
                for setting_name in setting.keys():
                    details = setting[setting_name]
                    description = None
                    value = None
                    for detail in details:
                        if ProjectConfiguration.setting_description in detail.keys():
                            description = detail[ProjectConfiguration.setting_description]
                        elif ProjectConfiguration.setting_value in detail.keys():
                            value = detail[ProjectConfiguration.setting_value]

                    lbl = Label(self.master_win, text=description)
                    lbl.grid(row=current_row, column=0, columnspan=1, sticky='nwse')
                    txt = Text(self.master_win, height=1, width=40, wrap="none")
                    txt.grid(row=current_row, column=1, columnspan=2, sticky='nwse', pady=10)
                    txt.insert(END, value)

                    self.settings[setting_name] = txt
                    current_row += 1

        # Add in the save/cancel buttons
        save_button = Button(self.master_win, text="Save", command=self.save_setting)
        save_button.grid(row=current_row, column=1, columnspan=1, sticky='nwse')
        close_button = Button(self.master_win, text="Cancel", command=self.cancel)
        close_button.grid(row=current_row, column=2, columnspan=1, sticky='nwse')

    def cancel(self):
        """ Cancel clicked, just close the window. """
        self.master_win.destroy()

    def save_setting(self):
        """
            Save clicked
                - For each row, collect the setting name and user input.
                    - Clean user input
                - Set values for all settings
                - Save configuration
                - Close window
        """
        for setting in self.settings.keys():
            user_entered = self.settings[setting].get("1.0", END)
            user_entered = user_entered.strip().replace('\n', '')
            print("Updating {} with '{}'".format(setting, user_entered))
            self.configuration.set_value(setting, user_entered)
        self.configuration.save_configuration()
        self.cancel()

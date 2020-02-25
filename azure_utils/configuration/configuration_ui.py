"""
- configuraitonui.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from tkinter import Frame, TRUE, Label, Text, END, Button
from tkinter import messagebox

from azure_utils.configuration.configuration_validation import Validation, ValidationResult
from azure_utils.configuration.project_configuration import ProjectConfiguration


class SettingsUpdate(Frame):
    """
        UI Wrapper for project configuration settings.

        Provide a configuration file as described in configuration.ProjectConfiguration.

        A UI is built using a grid where each row consists of:
            setting_description | Text control to show accept values

        Final row of the grid has a save and cancel button.

        Save updates the configuration file with any settings put on the UI.
    """

    def __init__(self, project_configuration, master):
        Frame.__init__(self, master=master)

        # self.configuration  = Instance of ProjectConfiguration and master
        # self.master_win     = Instance of Tk application.
        # self.settings       = Will be a dictionary where
        #                         key = Setting name
        #                         value = Text control

        self.configuration = project_configuration
        self.master_win = master
        self.settings = {}

        # Set up validator
        self.validator = Validation()

        # Set up some window options

        self.master_win.title(self.configuration.project_name())
        self.master_win.resizable(width=TRUE, height=TRUE)
        self.master_win.configure(padx=10, pady=10)

        # Populate the grid first with settings followed by the two buttons (cancel/save)
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
        """
            Cancel clicked, just close the window.
        """
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
        validate_responses = self.prompt_field_validation()
        field_responses = []

        for setting in self.settings:
            user_entered = self.settings[setting].get("1.0", END)
            user_entered = user_entered.strip().replace('\n', '')

            # Validate it
            if validate_responses:
                res = self.validator.validate_input(setting, user_entered)
                field_responses.append(res)
                Validation.dump_validation_result(res)
            else:
                print("Updating {} with '{}'".format(setting, user_entered))

            self.configuration.set_value(setting, user_entered)

        if self.validate_responses(field_responses):
            print("Writing out new configuration options...")
            self.configuration.save_configuration()
            self.cancel()

    @staticmethod
    def validate_responses(validation_responses) -> bool:
        """
        Determine if there are any failures or warnings. If so, give the user the
        option on staying on the screen to fix them.

        :param validation_responses: Response to validate
        :return: `bool` validation outcome
        """
        return_value = True

        if len(validation_responses) > 0:
            failed = [x for x in validation_responses if x.status == ValidationResult.failure]
            warn = [x for x in validation_responses if x.status == ValidationResult.warning]

            error_count = 0
            message = ""
            if failed:
                message += "ERRORS:\n"
                for resp in failed:
                    error_count += 1
                    message += "   {}\n".format(resp.type)
                message += '\n'

            if warn:
                message += "WARNINGS:\n"
                for resp in warn:
                    if resp.reason != Validation.FIELD_NOT_RECOGNIZED:
                        error_count += 1
                        message += "   {}:\n{}\n\n".format(resp.type, resp.reason)
                message += '\n'

            if error_count > 0:
                user_prefix = "The following fields either failed validation or produced a warning :\n\n"
                user_postfix = "Click Yes to continue with these validation issues or No to correct them."
                return_value = messagebox.askyesno('Validate Errors',
                                                   "{}{}{}".format(user_prefix, message, user_postfix))

        return return_value

    def prompt_field_validation(self) -> bool:
        """
        Prompt user for field to validation

        :return: `bool` based on user's response
        """
        valid_fields = "\n"
        for setting in self.settings:
            if self.validator.is_field_valid(setting):
                valid_fields += "{}\n".format(setting)

        user_prefix = "The following fields can be validated :\n\n"
        user_postfix = "\nValidation will add several seconds to the save, would you like to validate these settings?"

        return messagebox.askyesno('Validate Inputs', "{}{}{}".format(user_prefix, valid_fields, user_postfix))

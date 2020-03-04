"""
AI-Utilities - notebook_configuration_widget.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import yaml
from ipywidgets import widgets

from azure_utils.configuration.project_configuration import ProjectConfiguration


def get_configuration_widget(config):
    proj_config = ProjectConfiguration(config)
    proj_config.save_configuration()
    out = widgets.Output()

    uploader = widgets.FileUpload(
        accept='.yml',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
        multiple=False  # True to accept multiple files upload else False
    )
    save_upload = widgets.Button(
        description='Load Upload',
        disabled=False,
        button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click to save settings to file.',
        icon='check'
    )

    def on_button_clicked(_):
        if uploader.value:
            for file in uploader.value:
                with open(proj_config.configuration_file, 'wb') as f:
                    f.write(uploader.value[file]['content'])
                    f.close()
                with out:
                    print("Saved File: " + file)

    save_upload.on_click(on_button_clicked)

    subscription_id = widgets.Text(
        value=proj_config.get_value("subscription_id"),
        placeholder='xxxx-xxxx-xxxx-xxxx-xxxx',
        description='Sub ID:',
        disabled=False
    )
    workspace_name = widgets.Text(
        value=proj_config.get_value("workspace_name"),
        placeholder='azure-resource-group-name',
        description='RG ID:',
        tooltip='Click to save settings to file.',
        disabled=False
    )

    save = widgets.Button(
        description='Update Settings',
        disabled=False,
        button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click to save settings to file.',
        icon='check'
    )

    def on_button_clicked(_):
        # "linking function with output"
        proj_config.set_value("subscription_id", subscription_id.value)
        proj_config.set_value("workspace_name", workspace_name.value)
        proj_config.save_configuration()
        with open(proj_config.configuration_file, 'w') as f:
            f.write(yaml.safe_dump(proj_config.configuration))
            f.close()

    save.on_click(on_button_clicked)

    text_boxes = {}
    for setting in proj_config.get_settings():
        for setting_key in setting:
            text_boxes[setting_key] = widgets.Text(
                value=proj_config.get_value(setting_key).replace("<>", ""),
                placeholder=setting[setting_key][0]['description'],
                description=setting_key,
                disabled=False
            )

    my_list = [out, uploader, save_upload]
    for setting_key in text_boxes:
        my_list.append(text_boxes[setting_key])

    my_list.append(save)

    return widgets.VBox(my_list)

"""
AI-Utilities - notebook_configuration_widget.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import getpass
import warnings

import yaml
from azure.common.client_factory import get_client_from_cli_profile
from azure.mgmt.resource import SubscriptionClient
from azureml.core import Workspace
from ipywidgets import widgets
from knack.util import CLIError

from azure_utils.azureml_tools.subscription import run_az_cli_login
from azure_utils.configuration.project_configuration import ProjectConfiguration
from azure_utils.machine_learning.contexts.realtime_score_context import RealtimeScoreContext


def list_subscriptions():
    """

    :return:
    """
    try:
        sub_client = get_client_from_cli_profile(SubscriptionClient)
    except CLIError:
        run_az_cli_login()
        sub_client = get_client_from_cli_profile(SubscriptionClient)

    subs = sub_client.subscriptions.list()

    return [sub_map(subs, False), sub_map(subs)]


def sub_map(subs, id_or_name=True):
    if id_or_name:
        return {name2id(sub) for sub in subs}
    return {id2name(sub) for sub in subs}


def name2id(sub):
    return {sub.display_name: sub.subscription_id}


def id2name(sub):
    return {sub.subscription_id: sub.display_name}


def get_configuration_widget(config, with_existing=True):
    """

    :param config:
    :param with_existing:
    :return:
    """
    proj_config = ProjectConfiguration(config)
    proj_config.save_configuration()
    out = widgets.Output()

    uploader = widgets.FileUpload(accept='.yml', multiple=False)

    name2id, id2name = list_subscriptions()

    def update_and_save_configuration():
        """

        """
        for boxes in setting_boxs:
            proj_config.set_value(boxes, setting_boxs[boxes].value)
        proj_config.set_value("subscription_id", name2id[setting_boxs['subscription_id'].value])
        save_project_configuration()

    def save_project_configuration():
        """

        """
        with open(proj_config.configuration_file, 'w') as f:
            f.write(yaml.safe_dump(proj_config.configuration))
            f.close()

    getpass.getuser()

    setting_boxs = {}
    user_id = getpass.getuser()
    for setting in proj_config.get_settings():
        for setting_key in setting:
            setting_with_id = proj_config.get_value(setting_key).replace("$(User)", user_id)
            proj_config.set_value(setting_key, setting_with_id)

            setting = setting[setting_key][0]
            description = setting['description']

            setting_boxs[setting_key] = widgets.Text(value=setting_with_id.replace("<>", ""),
                                                     placeholder=description,
                                                     description=setting_key,
                                                     disabled=False)

    proj_config.save_configuration()

    default_sub = list(name2id.keys())[0]
    if proj_config.get_value('subscription_id') in id2name:
        default_sub = id2name[proj_config.get_value('subscription_id')]

    setting_boxs['subscription_id'] = widgets.Dropdown(options=list(name2id.keys()), value=default_sub,
                                                       description='subscription_id', disabled=False)

    def convert_to_region(key):
        """

        :param key:
        """
        if key in setting_boxs:
            setting_boxs[key] = widgets.Dropdown(
                options=['eastus', 'eastus2', 'canadacentral', 'centralus', 'northcentralus', 'southcentralus',
                         'westcentralus', 'westus', 'westus2'],
                value=proj_config.get_value(key).replace("<>", "eastus"), description=key, disabled=False)

    convert_to_region('workspace_region')
    convert_to_region('aks_location')
    convert_to_region('deep_aks_location')

    dropdown_keys = ["aks_service_name", "aks_name", "image_name",
                     "deep_aks_service_name", "deep_aks_name", "deep_image_name"]

    ws = Workspace(proj_config.get_value('subscription_id'),
                   proj_config.get_value('resource_group'),
                   proj_config.get_value('workspace_name'))

    ws = RealtimeScoreContext.get_or_create_workspace(config)

    def get_list(key):
        """

        :param key:
        :return:
        """
        if "image_name" in key:
            return list(ws.images.keys())
        if "aks_name" in key:
            return list(ws.compute_targets.keys())
        if "aks_service_name" in key:
            return list(ws.webservices.keys())

    my_list = [out, uploader]
    for setting_key in setting_boxs:
        text_box = setting_boxs[setting_key]
        if with_existing and setting_key in dropdown_keys and type(text_box) is widgets.Text:
            if get_list(setting_key)[0]:
                dropdown = widgets.Dropdown(options=get_list(setting_key), value=get_list(setting_key)[0],
                                            description='existing', disabled=False)
                text_box = widgets.HBox([text_box, dropdown])
        my_list.append(text_box)

    def upload_on_change(change):
        """

        :param change:
        """
        if change['type'] == 'change' and change['name'] == 'value':
            for file in uploader.value:
                with open(config, 'wb') as f:
                    f.write(uploader.value[file]['content'])
                    f.close()
                new_proj_config = ProjectConfiguration(config)

                for setting_box_key in setting_boxs:
                    if new_proj_config.has_value(setting_box_key):
                        if setting_box_key == "subscription_id":
                            setting_boxs[setting_box_key].value = id2name[new_proj_config.get_value(setting_box_key)]
                        else:
                            setting_boxs[setting_box_key].value = new_proj_config.get_value(setting_box_key)
                    else:
                        warnings.warn("Reload Widget to display new properties")

    uploader.observe(upload_on_change)

    def on_change(change):
        """

        :param change:
        """
        if change['type'] == 'change' and change['name'] == 'value':
            update_and_save_configuration()

    for box in setting_boxs:
        setting_boxs[box].observe(on_change)

    return widgets.VBox(my_list)

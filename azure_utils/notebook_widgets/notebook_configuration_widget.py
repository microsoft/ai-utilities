"""
AI-Utilities - notebook_configuration_widget.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import getpass
import warnings
from typing import List

import yaml
from azure.mgmt.resource import SubscriptionClient
from azureml.core import Workspace
from ipywidgets import widgets, VBox

from azure_utils.configuration.project_configuration import ProjectConfiguration
from azure_utils.machine_learning.contexts.realtime_score_context import (RealtimeScoreContext, )


def list_subscriptions(config) -> List[set]:
    """

    :return:
    """
    ws = RealtimeScoreContext.get_or_create_workspace(config)

    sub_client = SubscriptionClient(ws._auth)
    subs = sub_client.subscriptions.list()

    return [{name_2_id(sub) for sub in subs}, {id_2_name(sub) for sub in subs}]


def name_2_id(sub) -> dict:
    """
    Create Mapping for sub name to id
    :param sub: Tuple of one subscription
    :return: mapping_dict
    """
    return {sub.display_name: sub.subscription_id}


def id_2_name(sub) -> dict:
    """
    Create Mapping for id to sub name
    :param sub: Tuple of one subscription
    :return: mapping_dict
    """
    return {sub.subscription_id: sub.display_name}


def save_project_configuration(proj_config: ProjectConfiguration):
    """
    Thread-safe - Save project configuration
    """
    with open(proj_config.configuration_file, "w") as f:
        f.write(yaml.safe_dump(proj_config.configuration))
        f.close()


def update_and_save_configuration(
    proj_config: ProjectConfiguration, setting_boxes: dict, name2id: dict
):
    """
    Update project configuration with widget values and save

    :param proj_config: Project Configuration loaded from file.
    :param setting_boxes: Dict of Each Settings Widget
    :param name2id: Mapping of Subscription Name to ID
    """
    for boxes in setting_boxes:
        proj_config.set_value(boxes, setting_boxes[boxes].value)
    proj_config.set_value(
        "subscription_id", name2id[setting_boxes["subscription_id"].value]
    )
    save_project_configuration(proj_config)


def get_configuration_widget(config: str, with_existing: bool = True) -> VBox:
    """
    Get Configuration Widget for Configuration File

    :param config: project configuration filename
    :param with_existing:
    :return:
    """
    proj_config = ProjectConfiguration(config)
    proj_config.save_configuration()
    out = widgets.Output()

    uploader = widgets.FileUpload(accept=".yml", multiple=False)

    name2id = {'sub_id': proj_config.get_value("subscription_id")}
    id2name = {proj_config.get_value("subscription_id"): "sub_id"}
    getpass.getuser()

    default_sub = proj_config.get_value("subscription_id")

    setting_boxes = create_settings_boxes(default_sub, name2id, proj_config)

    proj_config.save_configuration()

    def convert_to_region(key: str) -> None:
        """

        :param key:
        """
        if key in setting_boxes:
            setting_boxes[key] = widgets.Dropdown(
                options=[
                    "eastus",
                    "eastus2",
                    "canadacentral",
                    "centralus",
                    "northcentralus",
                    "southcentralus",
                    "westcentralus",
                    "westus",
                    "westus2",
                ],
                value=proj_config.get_value(key).replace("<>", "eastus"),
                description=key,
                disabled=False,
            )

    convert_to_region("workspace_region")
    convert_to_region("aks_location")
    convert_to_region("deep_aks_location")

    dropdown_keys = [
        "aks_service_name",
        "aks_name",
        "image_name",
        "deep_aks_service_name",
        "deep_aks_name",
        "deep_image_name",
    ]

    ws = RealtimeScoreContext.get_or_create_workspace(config)

    my_list = get_widgets_list(
        dropdown_keys, out, setting_boxes, uploader, with_existing, ws
    )

    def upload_on_change(change: dict):
        """

        :param change:
        """
        if change["type"] == "change" and change["name"] == "value":
            for file in uploader.value:
                with open(config, "wb") as f:
                    f.write(uploader.value[file]["content"])
                    f.close()
                new_proj_config = ProjectConfiguration(config)

                update_setting_boxes(new_proj_config, setting_boxes, id2name)

    uploader.observe(upload_on_change)

    def on_change(change: dict):
        """

        :param change:
        """
        if change["type"] == "change" and change["name"] == "value":
            update_and_save_configuration(proj_config, setting_boxes, name2id)

    for box in setting_boxes:
        setting_boxes[box].observe(on_change)

    return widgets.VBox(my_list)


def get_widgets_list(
    dropdown_keys: list,
    out: widgets.Output,
    setting_boxes: dict,
    uploader: widgets.FileUpload,
    with_existing: bool,
    ws: Workspace,
) -> list:
    """
    Get Widgets in a List

    :param dropdown_keys: List of each of the option types to be listed in drop down
    :param out: Widget Output to add to list
    :param setting_boxes: list of each setting widget
    :param uploader: upload file widget to add to list
    :param with_existing: kernel existing flag
    :param ws: AzureML Workspace
    :return: List of all Widgets
    """
    my_list = [out, uploader]
    for setting_key in setting_boxes:
        text_box = setting_boxes[setting_key]
        is_valid = check_if_valid(
            dropdown_keys, setting_key, text_box, with_existing, ws
        )
        if is_valid:
            dropdown = widgets.Dropdown(
                options=get_list(ws, setting_key),
                value=get_list(ws, setting_key)[0],
                description="existing",
                disabled=False,
            )
            text_box = widgets.HBox([text_box, dropdown])
        my_list.append(text_box)
    return my_list


def check_if_valid(
    dropdown_keys: list,
    setting_key: str,
    text_box: widgets.Widget,
    with_existing: bool,
    ws: Workspace,
) -> bool:
    """
    Check if dropdown key list is valid.

    :param dropdown_keys: List of each of the option types to be listed in drop down
    :param setting_key: key of setting
    :param text_box: widget of setting
    :param with_existing: existing kernel flag
    :param ws: AzureML Workspace
    :return: `bool` result of check
    """
    return (
        with_existing
        and setting_key in dropdown_keys
        and type(text_box) is widgets.Text
        and get_list(ws, setting_key)[0]
    )


def create_settings_boxes(
    default_sub: str, name2id: dict, proj_config: ProjectConfiguration
) -> dict:
    """
    Create Settings Boxes

    :param default_sub: Sub to set as current selection
    :param name2id: sub name to id mapping
    :param proj_config: Project Configuration Container
    :return: Text Widgets for each setting.
    """
    setting_boxs = {}
    user_id = getpass.getuser()
    for setting in proj_config.get_settings():
        for setting_key in setting:
            setting_with_id = proj_config.get_value(setting_key).replace(
                "$(User)", user_id
            )
            proj_config.set_value(setting_key, setting_with_id)

            setting = setting[setting_key][0]
            description = setting["description"]

            setting_boxs[setting_key] = widgets.Text(
                value=setting_with_id.replace("<>", ""),
                placeholder=description,
                description=setting_key,
                disabled=False,
            )
    return setting_boxs


def get_list(ws: Workspace, key: str) -> list:
    """
    Get List of Workspace Resources based on key.

    :param ws: AzureML Workspace
    :param key: Service Type. ex: image_name, aks_name, aks_service_name
    :return: List of services in workspace
    """
    if "image_name" in key:
        return list(ws.images.keys())
    if "aks_name" in key:
        return list(ws.compute_targets.keys())
    if "aks_service_name" in key:
        return list(ws.webservices.keys())


def update_setting_boxes(
    new_proj_config: ProjectConfiguration, setting_boxes: dict, id2name: dict
):
    """
    Update Settings with new Project Configuration

    :param new_proj_config: Newly loaded project configuration
    :param setting_boxes: list of setting widgets
    :param id2name: mapping of sub ids to sub names
    """
    for setting_box_key in setting_boxes:
        if new_proj_config.has_value(setting_box_key):
            if setting_box_key == "subscription_id":
                setting_boxes[setting_box_key].value = id2name[
                    new_proj_config.get_value(setting_box_key)
                ]
            else:
                setting_boxes[setting_box_key].value = new_proj_config.get_value(
                    setting_box_key
                )
        else:
            warnings.warn("Reload Widget to display new properties")

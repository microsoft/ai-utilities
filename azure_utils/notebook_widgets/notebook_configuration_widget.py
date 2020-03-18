"""
AI-Utilities - notebook_configuration_widget.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import getpass
import warnings
from typing import List

import yaml
from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from ipywidgets import widgets, VBox, Layout

from azure_utils.configuration.project_configuration import ProjectConfiguration
from azure_utils.machine_learning.contexts.realtime_score_context import (
    RealtimeScoreContext,
)


def name_2_id(sub) -> dict:
    """
    Create Mapping for sub name to id
    :param sub: Tuple of one subscription
    :return: mapping_dict
    """
    return {sub.subscription_name: sub.subscription_id}


def id_2_name(sub) -> dict:
    """
    Create Mapping for id to sub name
    :param sub: Tuple of one subscription
    :return: mapping_dict
    """
    return {sub.subscription_id: sub.subscription_name}


def list_subscriptions():
    """	
    :return:	
    """

    auth = InteractiveLoginAuthentication()
    subs = Workspace._fetch_subscriptions(auth)[0]

    return [
        {sub.subscription_name: sub.subscription_id for sub in subs},
        {sub.subscription_id: sub.subscription_name for sub in subs},
    ]


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
        if boxes == "subscription_id":
            proj_config.set_value(boxes, name2id[setting_boxes[boxes].value])
        else:
            proj_config.set_value(boxes, setting_boxes[boxes].value)
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

    uploader = widgets.FileUpload(
        accept=".yml", multiple=False, layout=Layout(width="50%")
    )

    name2id, id2name = list_subscriptions()
    # name2id = {'sub_id': proj_config.get_value("subscription_id")}
    # id2name = {proj_config.get_value("subscription_id"): "sub_id"}
    getpass.getuser()

    # default_sub = proj_config.get_value("subscription_id")
    default_sub = list(name2id.keys())[0]
    if proj_config.get_value("subscription_id") in id2name:
        default_sub = id2name[proj_config.get_value("subscription_id")]
    else:
        proj_config.set_value("subscription_id", list(id2name.keys())[0])
        proj_config.save_configuration()

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
                style={"description_width": "initial"},
                layout=Layout(width="50%"),
            )

    convert_to_region("workspace_region")
    convert_to_region("aks_location")
    convert_to_region("deep_aks_location")

    dropdown_keys = []

    my_list = get_widgets_list(
        dropdown_keys, out, setting_boxes, uploader, with_existing, config
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
    config,
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
            dropdown_keys, setting_key, text_box, with_existing, config
        )
        if is_valid:

            ws = RealtimeScoreContext.get_or_create_workspace(config)

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
    config,
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
        and get_list(config, setting_key)[0]
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
            if setting_key == "subscription_id":
                setting_boxs["subscription_id"] = widgets.Dropdown(
                    options=list(name2id.keys()) + [""],
                    value=default_sub.replace("<>", ""),
                    description="subscription_id",
                    disabled=False,
                    style={"description_width": "initial"},
                    layout=Layout(width="50%"),
                )
            else:
                setting_boxs[setting_key] = widgets.Text(
                    value=setting_with_id.replace("<>", ""),
                    placeholder=description,
                    description=setting_key,
                    disabled=False,
                    style={"description_width": "initial"},
                    layout=Layout(width="50%"),
                )
    return setting_boxs


def get_list(config, key: str) -> list:
    """
    Get List of Workspace Resources based on key.

    :param ws: AzureML Workspace
    :param key: Service Type. ex: image_name, aks_name, aks_service_name
    :return: List of services in workspace
    """
    ws = RealtimeScoreContext.get_or_create_workspace(config)

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


class MockRequest:
    """Mock Request Class to create calls to test web service code"""

    method = "GET"


import ipywidgets as widgets
from IPython.display import display
import time
import threading


class BackgroundCountThread:
    def __init__(self, sleep_sec: float = 4):
        self._running = True
        self._sleep_sec = sleep_sec

    def terminate(self):
        self._running = False

    def work(self, slider):
        total = 100
        for i in range(total):
            time.sleep(self._sleep_sec)
            slider.value = i
            if not self._running:
                slider.value = 100
                break


def test_train_py_button(train_py="script/train_dl.py"):
    button = widgets.Button(
        description="Test train.py", layout=Layout(width="80%", height="80px"),
    )
    output = widgets.Output()
    slider = widgets.IntProgress(layout=Layout(width="80%"))

    def on_button_clicked(b):
        c = BackgroundCountThread(sleep_sec=2.4)
        slider.value = 0
        slider.bar_style = "info"
        thread = threading.Thread(target=c.work, args=(slider,))
        thread.start()
        try:
            with output:
                print("Train Started")
                button.disabled = True
                button.description = "Running"
                button.button_style = "info"

            exec(open(train_py).read())
            exec("train()")

            with output:
                print("Train Complete")
            button.button_style = "success"
            slider.bar_style = "success"
            button.description = "Complete, rerun?"

        except:
            with output:
                print("Score Test Error")
            button.button_style = "danger"
            slider.bar_style = "danger"
            button.description = "Error"
            raise
        finally:
            c.terminate()
            slider.value = 100
            button.disabled = False

    button.on_click(on_button_clicked)
    run_button = widgets.VBox([button, slider, output])
    return run_button


def test_score_py_button(score_py="source/score.py"):
    button = widgets.Button(
        description="Test score.py",
        layout=Layout(width="80%", height="80px", align_content="center"),
    )
    output = widgets.Output()
    slider = widgets.IntProgress(layout=Layout(width="80%"))

    def on_button_clicked(b):
        c = BackgroundCountThread(1)
        slider.value = 0
        slider.bar_style = "info"
        thread = threading.Thread(target=c.work, args=(slider,))
        thread.start()
        try:
            with output:
                print("Test Begin")
            button.disabled = True
            button.description = "Running"
            button.button_style = "info"

            exec(open(score_py).read())
            exec("init()")
            with output:
                exec("response = run(MockRequest())")
                exec("assert response")
                print("Score Test Complete")
            button.button_style = "success"
            slider.bar_style = "success"
            slider.value = 100
            button.description = "Complete, rerun?"

        except:
            with output:
                print("Score Test Error")
            button.button_style = "danger"
            slider.bar_style = "danger"
            button.description = "Error"
            raise
        finally:
            c.terminate()
            button.disabled = False

    button.on_click(on_button_clicked)
    run_button = widgets.VBox([button, slider, output])
    return run_button


def deploy_button(project_configuration, train_py="train_dl.py", score_py="score.py"):
    button = widgets.Button(
        description="Deploy Azure Machine Learning Services",
        layout=Layout(width="80%", height="80px", justify_content="center"),
    )
    output = widgets.Output()
    slider = widgets.IntProgress(layout=Layout(width="80%"))

    def on_button_clicked(b):
        c = BackgroundCountThread(12)
        slider.value = 0
        slider.bar_style = "info"
        thread = threading.Thread(target=c.work, args=(slider,))
        thread.start()
        try:
            button.disabled = True
            button.description = "Running"
            with output:
                button.button_style = "info"
                print("Begin Deployment.")
                from azure_utils.machine_learning.contexts.realtime_score_context import (
                    DeepRealtimeScore,
                )

                deep_ws, aks_service = DeepRealtimeScore.get_or_or_create(
                    configuration_file=project_configuration,
                    train_py=train_py,
                    score_py=score_py,
                )
                button.button_style = "success"
                button.description = "Complete, rerun?"
                print("Deploy Complete")
                display(deep_ws.workspace_widget)
        except:
            print("Deploy Error")
            button.button_style = "danger"
            slider.bar_style = "danger"
            button.description = "Error"
            raise
        finally:
            c.terminate()
            button.disabled = False

    button.on_click(on_button_clicked)
    run_button = widgets.VBox([button, slider, output])
    return run_button
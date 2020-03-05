"""
AI-Utilities - notebook_configuration_widget.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import getpass

import yaml
from azure.common.client_factory import get_client_from_cli_profile
from azure.mgmt.resource import SubscriptionClient
from ipywidgets import widgets
from knack.util import CLIError

from azure_utils.azureml_tools.subscription import _run_az_cli_login
from azure_utils.configuration.project_configuration import ProjectConfiguration


def list_subscriptions():
    try:
        sub_client = get_client_from_cli_profile(SubscriptionClient)
    except CLIError:
        _run_az_cli_login()
        sub_client = get_client_from_cli_profile(SubscriptionClient)

    return [{sub.display_name: sub.subscription_id for sub in sub_client.subscriptions.list()},
            {sub.subscription_id: sub.display_name for sub in sub_client.subscriptions.list()}]


def get_configuration_widget(config):
    proj_config = ProjectConfiguration(config)
    proj_config.save_configuration()
    out = widgets.Output()

    uploader = widgets.FileUpload(accept='.yml', multiple=False)
    # 'success', 'info', 'warning', 'danger' or ''
    save_upload = widgets.Button(description='Save Upload', disabled=False, button_style='',
                                 tooltip='Click to save settings to file.', icon='check')

    overwrite = widgets.Button(description='Overwrite Settings', disabled=False, button_style='',
                               tooltip='Click to save settings to file.', icon='check')
    name2id, id2name = list_subscriptions()

    def on_button_clicked(_):
        # "linking function with output"
        update_and_save_configuration()
        overwrite.button_style = 'success'

    def update_and_save_configuration():
        for boxes in text_boxes:
            proj_config.set_value(boxes, text_boxes[boxes].value)
        proj_config.set_value("subscription_id", name2id[text_boxes['subscription_id'].value])
        save_project_configuration()

    def save_project_configuration():
        with open(proj_config.configuration_file, 'w') as f:
            f.write(yaml.safe_dump(proj_config.configuration))
            f.close()

    overwrite.on_click(on_button_clicked)

    getpass.getuser()

    text_boxes = {}
    user_id = getpass.getuser()
    for setting in proj_config.get_settings():
        for setting_key in setting:
            setting_with_id = proj_config.get_value(setting_key).replace("$(User)", user_id)
            proj_config.set_value(setting_key, setting_with_id)

            text = widgets.Text(value=setting_with_id.replace("<>", ""),
                                placeholder=setting[setting_key][0]['description'], description=setting_key,
                                disabled=False)

            text_boxes[setting_key] = widgets.HBox([text])
    proj_config.save_configuration()

    default_sub = list(name2id.keys())[0]
    if proj_config.get_value('subscription_id') in id2name:
        default_sub = id2name[proj_config.get_value('subscription_id')]

    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            update_and_save_configuration()

    text_boxes['subscription_id'] = widgets.Dropdown(options=list(name2id.keys()), value=default_sub,
                                                     description='subscription_id', disabled=False)
    text_boxes['subscription_id'].observe(on_change)

    def convert_to_region(key):
        if key in text_boxes:
            text_boxes[key] = widgets.Dropdown(
                options=['eastus', 'eastus2', 'canadacentral', 'centralus', 'northcentralus', 'southcentralus',
                         'westcentralus', 'westus', 'westus2'],
                value=proj_config.get_value(key).replace("<>", "eastus"), description=key, disabled=False)
            text_boxes[key].observe(on_change)

    convert_to_region('workspace_region')
    convert_to_region('aks_location')
    convert_to_region('deep_aks_location')

    my_list = [out, widgets.HBox([uploader, save_upload])]
    for setting_key in text_boxes:
        my_list.append(text_boxes[setting_key])

    my_list.append(overwrite)

    def on_upload_save_click(_):
        if uploader.value:
            for file in uploader.value:
                with open(config, 'wb') as f:
                    f.write(uploader.value[file]['content'])
                    f.close()
                proj_config = ProjectConfiguration(config)
                for key in text_boxes:
                    assert proj_config.has_value(key)
                    if key != "subscription_id":
                        text_boxes[key].value = proj_config.get_value(key)
                    else:
                        text_boxes[key].value = id2name[proj_config.get_value(key)]

                save_upload.button_style = "success"

    save_upload.on_click(on_upload_save_click)

    return widgets.VBox(my_list)


def make_workspace_widget(model_dict, aks_dict):
    from ipywidgets import widgets
    def make_vbox(model_dict):
        labels = []
        for k in model_dict:
            if type(model_dict[k]) is not dict:
                string = str(model_dict[k])
                labels.append(widgets.HBox([widgets.HTML(value="<b>" + k + ":</b>"), widgets.Label(string)]))
            else:
                mini_labels = []
                mini_dic = model_dict[k]
                if mini_dic and mini_dic is not dict:
                    for mini_k in mini_dic:
                        string = str(mini_dic[mini_k])
                        mini_labels.append(
                            widgets.HBox([widgets.HTML(value="<b>" + mini_k + ":</b>"), widgets.Label(string)]))
                    mini_model_accordion = widgets.Accordion(children=[widgets.VBox(mini_labels)])
                    mini_model_accordion.set_title(0, k)
                    labels.append(mini_model_accordion)

        model_widget = widgets.VBox(labels)
        return widgets.VBox(children=[model_widget])

    ws_image = widgets.HTML(
        value='<img src="https://raw.githubusercontent.com/microsoft/AI-Utilities/master/docs/studio.png">')
    model_vbox = make_vbox(model_dict)
    aks_box = make_vbox(aks_dict)

    deployment_accordion = widgets.Accordion(children=[ws_image, model_vbox])
    deployment_accordion.set_title(0, 'Workspace')
    deployment_accordion.set_title(1, 'Model')

    application_insights_images = [
        widgets.HTML(
            value='<img src="https://raw.githubusercontent.com/microsoft/AI-Utilities/master/docs/app_insights_1.png'
                  '">'),
        widgets.HTML(
            value='<img src="https://raw.githubusercontent.com/microsoft/AI-Utilities/master/docs'
                  '/app_insights_availability.png">'),
        widgets.HTML(
            value='<img src="https://raw.githubusercontent.com/microsoft/AI-Utilities/master/docs'
                  '/app_insights_perf_dash.png">'),
        widgets.HTML(
            value='<img src="https://raw.githubusercontent.com/microsoft/AI-Utilities/master/docs/app_insights_perf'
                  '.png">')
    ]
    application_insights_accordion = widgets.Accordion(children=application_insights_images)
    application_insights_accordion.set_title(0, 'Main')
    application_insights_accordion.set_title(1, 'Availability')
    application_insights_accordion.set_title(2, 'Performance')
    application_insights_accordion.set_title(3, 'Load Testing')

    kubernetes_image = widgets.HTML(
        value='<img src="https://raw.githubusercontent.com/microsoft/AI-Utilities/master/docs/kubernetes.png">')
    kubernetes_accordion = widgets.Accordion(children=[aks_box, kubernetes_image])
    kubernetes_accordion.set_title(0, 'Main')
    kubernetes_accordion.set_title(1, 'Performance')

    tab_nest = widgets.Tab()
    tab_nest.children = [deployment_accordion, kubernetes_accordion, application_insights_accordion]
    tab_nest.set_title(0, 'ML Studio')
    tab_nest.set_title(1, 'Kubernetes')
    tab_nest.set_title(2, 'Application Insights')
    return tab_nest

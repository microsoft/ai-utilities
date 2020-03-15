"""
AI-Utilities - azure_utils/notebook_widgets/workspace_widget.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from ipywidgets import widgets


def make_vbox(model_dict: dict) -> widgets.VBox:
    """

    :param model_dict:
    :return:
    """
    labels = []
    for k in model_dict:
        if type(model_dict[k]) is not dict:
            string = str(model_dict[k])
            labels.append(make_setting_hbox(k, string))
        else:
            mini_labels = []
            mini_dic = model_dict[k]
            if mini_dic and mini_dic is not dict:
                for mini_k in mini_dic:
                    string = str(mini_dic[mini_k])
                    mini_labels.append(make_setting_hbox(mini_k, string))
                mini_model_accordion = widgets.Accordion(
                    children=[widgets.VBox(mini_labels)]
                )
                mini_model_accordion.set_title(0, k)
                labels.append(mini_model_accordion)

    model_widget = widgets.VBox(labels)
    return widgets.VBox(children=[model_widget])


def make_setting_hbox(mini_k: str, string: str) -> widgets.HBox:
    """

    :param mini_k:
    :param string:
    :return:
    """
    return widgets.HBox(
        [widgets.HTML(value="<b>" + mini_k + ":</b>"), widgets.Label(string)]
    )


def make_workspace_widget(model_dict: dict, aks_dict: dict) -> widgets.Widget:
    """

    :param model_dict:
    :param aks_dict:
    :return:
    """

    ws_image = widgets.HTML(
        value='<img src="https://raw.githubusercontent.com/microsoft/AI-Utilities/master/docs/studio.png">'
    )
    model_vbox = make_vbox(model_dict)
    aks_box = make_vbox(aks_dict)

    deployment_accordion = widgets.Accordion(children=[ws_image, model_vbox])
    deployment_accordion.set_title(0, "Workspace")
    deployment_accordion.set_title(1, "Model")

    application_insights_images = [
        widgets.HTML(
            value='<img src="https://raw.githubusercontent.com/microsoft/AI-Utilities/master/docs/app_insights_1.png'
            '">'
        ),
        widgets.HTML(
            value='<img src="https://raw.githubusercontent.com/microsoft/AI-Utilities/master/docs'
            '/app_insights_availability.png">'
        ),
        widgets.HTML(
            value='<img src="https://raw.githubusercontent.com/microsoft/AI-Utilities/master/docs'
            '/app_insights_perf_dash.png">'
        ),
        widgets.HTML(
            value='<img src="https://raw.githubusercontent.com/microsoft/AI-Utilities/master/docs/app_insights_perf'
            '.png">'
        ),
    ]
    application_insights_accordion = widgets.Accordion(
        children=application_insights_images
    )
    application_insights_accordion.set_title(0, "Main")
    application_insights_accordion.set_title(1, "Availability")
    application_insights_accordion.set_title(2, "Performance")
    application_insights_accordion.set_title(3, "Load Testing")

    kubernetes_image = widgets.HTML(
        value='<img src="https://raw.githubusercontent.com/microsoft/AI-Utilities/master/docs/kubernetes.png">'
    )
    kubernetes_accordion = widgets.Accordion(children=[aks_box, kubernetes_image])
    kubernetes_accordion.set_title(0, "Main")
    kubernetes_accordion.set_title(1, "Performance")

    tab_nest = widgets.Tab()
    tab_nest.children = [
        deployment_accordion,
        kubernetes_accordion,
        application_insights_accordion,
    ]
    tab_nest.set_title(0, "ML Studio")
    tab_nest.set_title(1, "Kubernetes")
    tab_nest.set_title(2, "Application Insights")
    return tab_nest

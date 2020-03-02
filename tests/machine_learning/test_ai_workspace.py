"""
AI-Utilities - ${FILE_NAME}

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import os

from azure_utils.machine_learning.ai_workspace import DeepRTSWorkspace, RTSWorkspace


def test_get_or_create():
    ws = RTSWorkspace.get_or_create_workspace()
    assert type(ws) is RTSWorkspace
    assert ws


def test_get_or_create_deep_workspace():
    ws = DeepRTSWorkspace.get_or_create_workspace()
    assert type(ws) is DeepRTSWorkspace
    assert ws


def test_create_stack_overflow_data():
    ws = RTSWorkspace.get_or_create_workspace()
    questions, dupes_test = ws.get_or_create_data()
    print(questions)


def test_get_or_create_model():
    if not os.path.isfile("script/create_model.py"):
        os.makedirs("script", exist_ok=True)

        create_model_py = "from azure_utils.machine_learning import create_model\n\nif __name__ == '__main__':\n    " \
                          "create_model.main()"
        with open("script/create_model.py", "w") as file:
            file.write(create_model_py)

    ws = DeepRTSWorkspace.get_or_create_workspace()
    model = ws.get_or_create_model()
    assert model


def test_get_or_create_image():
    ws = RTSWorkspace.get_or_create_workspace()
    assert ws
    models = [ws.get_or_create_model()]
    assert models
    config = ws.get_or_create_image_configuration()
    assert config
    image = RTSWorkspace.get_or_create_image(config, "image_name", models)
    assert image


def test_get_or_create_aks():
    ws = RTSWorkspace.get_or_create_workspace()
    aks = ws.get_or_create_aks()
    assert aks


def test_get_or_create_service():
    ws = RTSWorkspace.get_or_create_workspace()
    service = ws.get_or_create_service()
    assert service


def test_get_or_create_deep_model():
    ws = DeepRTSWorkspace.get_or_create_workspace()
    ws.get_or_create_model()


def test_get_or_create_model_driver():
    ws = DeepRTSWorkspace.get_or_create_workspace()
    ws.get_or_create_model_driver()


def test_get_or_create_deep_image():
    ws = DeepRTSWorkspace.get_or_create_workspace()
    models = [ws.get_or_create_model()]
    config = ws.get_or_create_image_configuration()
    DeepRTSWorkspace.get_or_create_image(config, "deep_image_name", models=models)


def test_get_or_create_deep_service():
    ws = DeepRTSWorkspace.get_or_create_workspace()
    ws.get_or_create_service()

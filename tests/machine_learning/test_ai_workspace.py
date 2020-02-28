"""
AI-Utilities - ${FILE_NAME}

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import os

from azure_utils.machine_learning.ai_workspace import AILabWorkspace, DeepRTSWorkspace, RTSWorkspace


def test_get_or_create():
    ws = RTSWorkspace.get_or_create_workspace()
    assert ws


def test_create_stack_overflow_data():
    questions, dupes_test = RTSWorkspace.create_stack_overflow_data()
    print(questions)


def test_train_local():
    if not os.path.isfile("script/create_model.py"):
        os.makedirs("script", exist_ok=True)

        create_model_py = "from azure_utils.machine_learning import create_model\n\nif __name__ == '__main__':\n    " \
                          "create_model.main()"
        with open("script/create_model.py", "w") as file:
            file.write(create_model_py)

    model = RTSWorkspace.train_local()
    assert model


def test_get_or_create_image():
    image = RTSWorkspace.get_or_create_image()
    assert image


def test_get_or_create_aks():
    aks = RTSWorkspace.get_or_create_aks()
    assert aks


def test_get_or_create_service():
    service = RTSWorkspace.get_or_create_service()
    assert service


def test_get_or_create_deep_model():
    DeepRTSWorkspace.get_or_create_model()


def test_get_or_create_model_driver():
    DeepRTSWorkspace.get_or_create_model_driver()


def test_get_or_create_deep_image():
    DeepRTSWorkspace.get_or_create_image()


def test_get_or_create_deep_service():
    DeepRTSWorkspace.get_or_create_service()

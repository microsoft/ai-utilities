"""
AI-Utilities - ${FILE_NAME}

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from azure_utils.machine_learning.ai_workspace import DeepRTSWorkspace, RTSWorkspace


class TestDeployRTS:
    workspace_type = RTSWorkspace
    ws = workspace_type.get_or_create_workspace()
    image_setting_name = "image_name"

    def test_get_or_create(self):
        assert type(self.ws) is self.workspace_type
        assert self.ws

    def test_get_or_create_model(self):
        model = self.ws.get_or_create_model()
        assert model

    def test_get_or_create_image(self):
        models = [self.ws.get_or_create_model()]
        assert models
        config = self.ws.get_or_create_image_configuration()
        assert config
        image = self.ws.get_or_create_image(config, self.image_setting_name, models)
        assert image

    def test_get_or_create_aks(self):
        aks = self.ws.get_or_create_aks()
        assert aks

    def test_get_or_create_service(self):
        service = self.ws.get_or_create_service()
        assert service


class TestDeployDeepRTS(TestDeployRTS):
    workspace_type = DeepRTSWorkspace
    ws = workspace_type.get_or_create_workspace()
    image_setting_name = "deep_image_name"


def test_get_or_create_function_endpoint():
    RTSWorkspace.get_or_or_create_function_endpoint()

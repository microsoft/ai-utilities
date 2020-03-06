"""
AI-Utilities - ${FILE_NAME}

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import pytest

from azure_utils.machine_learning.contexts.realtime_score_context import DeepRealtimeScore, MLRealtimeScore
from deprecated import deprecated

class WorkspaceCreationTests:
    @pytest.fixture(scope="class")
    def workspace_type(self):
        return DeepRealtimeScore

    @pytest.fixture(scope="class")
    def realtime_score_workspace(self, workspace_type):
        return workspace_type.get_or_create_workspace(train_py="create_deep_model.py", score_py="driver.py")

    def test_get_or_create(self, realtime_score_workspace, workspace_type):
        assert type(realtime_score_workspace) is workspace_type
        assert realtime_score_workspace

    def test_get_images(self, realtime_score_workspace):
        assert realtime_score_workspace.images

    def test_get_compute_targets(self, realtime_score_workspace):
        assert realtime_score_workspace.compute_targets

    def test_get_webserices(self, realtime_score_workspace):
        assert realtime_score_workspace.webservices


class TestDeployRTS(WorkspaceCreationTests):
    @pytest.fixture(scope="class")
    def workspace_type(self):
        return MLRealtimeScore

    @pytest.fixture(scope="class")
    def realtime_score_workspace(self, workspace_type):
        return workspace_type.get_or_create_workspace(train_py="create_deep_model.py", score_py="driver.py")

    def test_get_or_create_model(self, realtime_score_workspace):
        assert realtime_score_workspace.get_or_create_model()

    @deprecated(version='0.3.81', reason="Switch to using Env, this will be removed in 0.4.0")
    def test_get_or_create_image(self, realtime_score_workspace):
        models = [realtime_score_workspace.get_or_create_model()]
        assert models
        config = realtime_score_workspace.get_or_create_image_configuration()
        assert config
        assert realtime_score_workspace.get_or_create_image(config, models)

    def test_get_or_create_aks(self, realtime_score_workspace):
        assert realtime_score_workspace.get_or_create_aks()

    def test_get_or_create_service(self, realtime_score_workspace):
        aks = realtime_score_workspace.get_or_create_aks()
        assert realtime_score_workspace.get_or_create_aks_service_with_image(aks)


class TestDeployDeepRTS(WorkspaceCreationTests):
    @pytest.fixture(scope="class")
    def workspace_type(self):
        return DeepRealtimeScore

    @pytest.fixture(scope="class")
    def realtime_score_workspace(self, workspace_type):
        return workspace_type.get_or_create_workspace(train_py="create_deep_model.py", score_py="driver.py")

    def test_get_or_create_model(self, realtime_score_workspace):
        assert realtime_score_workspace.get_or_create_model()

    def test_get_or_create_image(self, realtime_score_workspace):
        model = realtime_score_workspace.get_or_create_model()
        config = realtime_score_workspace.get_or_create_image_configuration()
        assert config
        assert realtime_score_workspace.get_or_create_image(config, [model])

    def test_get_or_create_aks(self, realtime_score_workspace):
        assert realtime_score_workspace.get_or_create_aks()

    def dont_test_get_or_create_service_with_image(self, realtime_score_workspace):
        aks = realtime_score_workspace.get_or_create_aks()
        assert realtime_score_workspace.get_or_create_aks_service_with_image(aks)

    def test_get_or_create_service(self, realtime_score_workspace):
        model = realtime_score_workspace.get_or_create_model()
        inference_config = realtime_score_workspace.get_inference_config()
        aks_target = realtime_score_workspace.get_or_create_aks()

        assert realtime_score_workspace.get_or_create_aks_service(model, aks_target, inference_config)


class TestDeployDeepRTSLocally:
    def test_train_py(self):
        import os
        if not os.path.isdir("outputs"):
            os.mkdir("outputs")
        os.system("python create_deep_model_new.py")
        import os.path
        assert os.path.isfile("outputs/model.pkl")

    def test_score_py(self):
        from tests.machine_learning.driver import init, run
        init()
        response = run(MockRequest())
        assert response


class MockRequest:
    method = 'GET'


def test_get_or_create_function_endpoint():
    MLRealtimeScore.get_or_or_create_function_endpoint()

"""
AI-Utilities - ${FILE_NAME}

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import pytest

from azure_utils.machine_learning.ai_workspace import DeepRealtimeScore, MLRealtimeScore


class TestDeployRTS:
    @pytest.fixture(scope="class")
    def ml_workspace_type(self):
        return MLRealtimeScore

    @pytest.fixture(scope="class")
    def ml_realtime_score_workspace(self, ml_workspace_type):
        return ml_workspace_type.get_or_create_workspace(train_py="create_deep_model.py", score_py="driver.py")

    def test_get_or_create(self, ml_realtime_score_workspace, ml_workspace_type):
        assert type(ml_realtime_score_workspace) is ml_workspace_type
        assert ml_realtime_score_workspace

    def test_get_or_create_model(self, ml_realtime_score_workspace):
        assert ml_realtime_score_workspace.get_or_create_model()

    def test_get_or_create_image(self, ml_realtime_score_workspace):
        models = [ml_realtime_score_workspace.get_or_create_model()]
        assert models
        config = ml_realtime_score_workspace.get_or_create_image_configuration()
        assert config
        assert ml_realtime_score_workspace.get_or_create_image(config, models)

    def test_get_or_create_aks(self, ml_realtime_score_workspace):
        assert ml_realtime_score_workspace.get_or_create_aks()

    def test_get_or_create_service(self, ml_realtime_score_workspace):
        aks = ml_realtime_score_workspace.get_or_create_aks()
        assert ml_realtime_score_workspace.get_or_create_aks_service(aks)


class TestDeployDeepRTS:
    @pytest.fixture(scope="class")
    def dl_workspace_type(self):
        return DeepRealtimeScore

    @pytest.fixture(scope="class")
    def deep_realtime_score_workspace(self, dl_workspace_type):
        return dl_workspace_type.get_or_create_workspace(train_py="create_deep_model.py", score_py="driver.py")

    def test_get_or_create(self, deep_realtime_score_workspace, dl_workspace_type):
        assert type(deep_realtime_score_workspace) is dl_workspace_type
        assert deep_realtime_score_workspace

    def test_get_or_create_model(self, deep_realtime_score_workspace):
        assert deep_realtime_score_workspace.get_or_create_model()

    def test_get_or_create_image(self, deep_realtime_score_workspace):
        model = deep_realtime_score_workspace.get_or_create_model()
        config = deep_realtime_score_workspace.get_or_create_image_configuration()
        assert config
        assert deep_realtime_score_workspace.get_or_create_image(config, [model])

    def test_get_or_create_aks(self, deep_realtime_score_workspace):
        assert deep_realtime_score_workspace.get_or_create_aks()

    def test_get_or_create_service(self, deep_realtime_score_workspace):
        aks = deep_realtime_score_workspace.get_or_create_aks()
        assert deep_realtime_score_workspace.get_or_create_aks_service(aks)

    def test_get_or_create_service_2(self, deep_realtime_score_workspace):
        from datetime import datetime
        print(datetime.now())
        model = deep_realtime_score_workspace.get_or_create_model()
        print(datetime.now())
        inference_config = deep_realtime_score_workspace.get_inference_config()
        print(datetime.now())
        aks_target = deep_realtime_score_workspace.get_or_create_aks()
        print(datetime.now())
        web_service = deep_realtime_score_workspace.get_or_create_aks_service_2(model, aks_target, inference_config)
        print(datetime.now())
        assert web_service


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

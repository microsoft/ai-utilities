"""
AI-Utilities - test_realtime_contexts

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import pytest
from deprecated import deprecated

from azure_utils.machine_learning.contexts.model_management_context import ModelManagementContext
from azure_utils.machine_learning.contexts.realtime_score_context import DeepRealtimeScore, MLRealtimeScore, \
    RealtimeScoreAKSContext, RealTimeScoreImageAndAKSContext
from azure_utils.machine_learning.contexts.workspace_contexts import WorkspaceContext


class WorkspaceCreationTests:
    """Workspace Creation Test Suite"""

    @pytest.fixture(scope="class")
    def context_type(self):
        """
        Abstract Workspace Type Fixture - Update with Workspace Context to test
        """
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def realtime_score_context(self, context_type: WorkspaceContext) -> WorkspaceContext:
        """
        Get or Create Context for Testing
        :param context_type: impl of WorkspaceContext
        :return:
        """
        return context_type.get_or_create_workspace(train_py="create_deep_model.py", score_py="driver.py")

    @staticmethod
    def test_get_or_create(realtime_score_context: RealtimeScoreAKSContext, context_type: WorkspaceContext):
        """
        Assert Context Type and Creation

        :param realtime_score_context: Testing Context
        :param context_type: Expected Context Type
        """
        assert type(realtime_score_context) is context_type
        assert realtime_score_context

    @staticmethod
    def test_get_images(realtime_score_context: RealtimeScoreAKSContext):
        """
        Assert images have been created

        :param realtime_score_context: Testing Context
        """
        assert realtime_score_context.images

    @staticmethod
    def test_get_compute_targets(realtime_score_context: RealtimeScoreAKSContext):
        """

        :param realtime_score_context: Testing Context
        """
        assert realtime_score_context.compute_targets

    @staticmethod
    def test_get_webserices(realtime_score_context: RealtimeScoreAKSContext):
        """

        :param realtime_score_context: Testing Context
        """
        assert realtime_score_context.webservices

    @pytest.fixture(scope="class")
    def test_files(self):
        """

        :return:
        """
        return {"train_py": "train.py", "score_py": "score.py"}

    @staticmethod
    def test_get_or_create_model(realtime_score_context: ModelManagementContext):
        """

        :param realtime_score_context: Testing Context
        """
        assert realtime_score_context.get_or_create_model()

    @deprecated(version='0.3.81', reason="Switch to using Env, this will be removed in 0.4.0")
    def test_get_or_create_image(self, realtime_score_context: RealtimeScoreAKSContext):
        """

        :param realtime_score_context: Testing Context
        """

        models = [RealTimeScoreImageAndAKSContext(realtime_score_context).get_or_create_model()]
        assert models
        config = RealTimeScoreImageAndAKSContext(realtime_score_context).get_inference_config()
        assert config
        assert RealTimeScoreImageAndAKSContext(realtime_score_context).get_or_create_image(config, models, )

    @staticmethod
    def test_get_or_create_aks(realtime_score_context: RealtimeScoreAKSContext):
        """

        :param realtime_score_context: Testing Context
        """
        assert realtime_score_context.get_or_create_aks()

    @staticmethod
    def test_get_or_create_service(realtime_score_context: RealtimeScoreAKSContext):
        """

        :param realtime_score_context: Testing Context
        """
        aks = RealTimeScoreImageAndAKSContext(realtime_score_context).get_or_create_aks()
        assert RealTimeScoreImageAndAKSContext(realtime_score_context).get_or_create_aks_service_with_image(aks)


class TestDeployRTS(WorkspaceCreationTests):

    @pytest.fixture(scope="class")
    def context_type(self):
        """

        :return:
        """
        return MLRealtimeScore


class TestDeployDeepRTS(WorkspaceCreationTests):
    @pytest.fixture(scope="class")
    def context_type(self):
        """

        :return:
        """
        return DeepRealtimeScore

    @pytest.fixture(scope="class")
    def test_files(self):
        return {"train_py": "train_dl.py", "score_py": "score_dl.py"}


# noinspection PyUnresolvedReferences,PyUnresolvedReferences,PyUnresolvedReferences
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
    """ MOCK request to test calling scoring service"""
    method = 'GET'


def test_get_or_create_function_endpoint():
    """Test creation of Azure Function for ML Scoring"""
    MLRealtimeScore.get_or_or_create_function_endpoint()

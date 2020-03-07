"""
AI-Utilities - test_realtime_contexts

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import pytest
from deprecated import deprecated

from azure_utils.machine_learning.contexts.model_management_context import ModelManagementContext
from azure_utils.machine_learning.contexts.realtime_score_context import DeepRealtimeScore, MLRealtimeScore, \
    RealtimeScoreAKSContext, RealTimeScoreImageAndAKSContext, RealtimeScoreFunctionsContext
from azure_utils.machine_learning.contexts.workspace_contexts import WorkspaceContext


# noinspection PyMethodMayBeStatic
class WorkspaceCreationTests:
    """Workspace Creation Test Suite"""

    @pytest.fixture(scope="class")
    def context_type(self):
        """
        Abstract Workspace Type Fixture - Update with Workspace Context to test
        """
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def realtime_score_context(self, context_type: RealtimeScoreAKSContext) -> RealtimeScoreAKSContext:
        """
        Get or Create Context for Testing
        :param context_type: impl of WorkspaceContext
        :return:
        """
        return context_type.get_or_create_workspace(train_py="create_deep_model.py", score_py="driver.py")

    def test_get_or_create(self, realtime_score_context: RealtimeScoreAKSContext, context_type: WorkspaceContext):
        """
        Assert Context Type and Creation

        :param realtime_score_context: Testing Context
        :param context_type: Expected Context Type
        """
        assert type(realtime_score_context) is context_type
        assert realtime_score_context

    def test_get_images(self, realtime_score_context: RealtimeScoreAKSContext):
        """
        Assert images have been created

        :param realtime_score_context: Testing Context
        """
        assert realtime_score_context.images

    def test_get_compute_targets(self, realtime_score_context: RealtimeScoreAKSContext):
        """

        :param realtime_score_context: Testing Context
        """
        assert realtime_score_context.compute_targets

    def test_get_webserices(self, realtime_score_context: RealtimeScoreAKSContext):
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

    def test_get_or_create_model(self, realtime_score_context: ModelManagementContext):
        """

        :param realtime_score_context: Testing Context
        """
        assert realtime_score_context.get_or_create_model()

    def test_get_or_create_aks(self, realtime_score_context: RealtimeScoreAKSContext):
        """

        :param realtime_score_context: Testing Context
        """
        assert realtime_score_context.get_or_create_aks()


class TestDeployRTS(WorkspaceCreationTests):

    @pytest.fixture(scope="class")
    def context_type(self):
        """

        :return:
        """
        return MLRealtimeScore

    @pytest.fixture(scope="class")
    def realtime_score_context(self, context_type: MLRealtimeScore) -> MLRealtimeScore:
        """
        Get or Create Context for Testing
        :param context_type: impl of WorkspaceContext
        :return:
        """
        return context_type.get_or_create_workspace(train_py="create_deep_model.py", score_py="driver.py")


class TestDeployDeepRTS(WorkspaceCreationTests):
    @pytest.fixture(scope="class")
    def context_type(self):
        """

        :return:
        """
        return DeepRealtimeScore

    @pytest.fixture(scope="class")
    def realtime_score_context(self, context_type: DeepRealtimeScore) -> DeepRealtimeScore:
        """
        Get or Create Context for Testing
        :param context_type: impl of WorkspaceContext
        :return:
        """
        return context_type.get_or_create_workspace(train_py="create_deep_model.py", score_py="driver.py")

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
    RealtimeScoreFunctionsContext.get_or_or_create_function_endpoint()

import os

import pytest
from azureml.core import Model

from azure_utils.configuration.notebook_config import project_configuration_file
from azure_utils.configuration.project_configuration import ProjectConfiguration
from azure_utils.machine_learning.contexts.realtime_score_context import (
    RealtimeScoreAKSContext,
    MLRealtimeScore,
    DeepRealtimeScore,
)
from azure_utils.machine_learning.contexts.workspace_contexts import WorkspaceContext
from tests.mocks.azureml.azureml_mocks import MockMLRealtimeScore, MockDeepRealtimeScore


class MockWorkspaceCreationTests:
    """Workspace Creation Test Suite"""

    @pytest.fixture(scope="class")
    def context_type(self):
        """
        Abstract Workspace Type Fixture - Update with Workspace Context to test
        """
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def files_for_testing(self):
        """

        :return:
        """
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def realtime_score_context(
        self, context_type: RealtimeScoreAKSContext, files_for_testing
    ) -> RealtimeScoreAKSContext:
        """
        Get or Create Context for Testing
        :param context_type: impl of WorkspaceContext
        :param test_files: Dict of input Files
        :return:
        """
        raise NotImplementedError

    def test_mock_get_or_create(
        self,
        realtime_score_context: RealtimeScoreAKSContext,
        context_type: WorkspaceContext,
    ):
        """
        Assert Context Type and Creation

        :param realtime_score_context: Testing Context
        :param context_type: Expected Context Type
        """
        assert realtime_score_context
        assert hasattr(realtime_score_context, "_subscription_id")
        assert hasattr(realtime_score_context, "_resource_group")
        assert hasattr(realtime_score_context, "_workspace_name")
        assert hasattr(realtime_score_context, "project_configuration_file")
        assert hasattr(realtime_score_context, "score_py")
        assert hasattr(realtime_score_context, "train_py")

    def test_mock_get_or_create_model(
        self, monkeypatch, realtime_score_context: MLRealtimeScore
    ):
        """

        :param realtime_score_context: Testing Context
        """

        @staticmethod
        def mockreturn_2(
            workspace, name, id, tags, properties, version, model_framework, run_id
        ):
            return {
                "name": "mock",
                "id": "1",
                "createdTime": "11/8/2020",
                "description": "",
                "mimeType": "a",
                "properties": "",
                "unpack": "",
                "url": "localhost",
                "version": 1,
                "experimentName": "expName",
                "runId": 1,
                "datasets": None,
                "createdBy": "mock",
                "framework": "python",
                "frameworkVersion": "1",
            }

        def mock_get_model_path_remote(model_name, version, workspace):
            return "."

        def mock_initialize(self, workspace, obj_dict):
            pass

        monkeypatch.setattr(Model, "_get", mockreturn_2)
        monkeypatch.setattr(Model, "_get_model_path_remote", mock_get_model_path_remote)
        monkeypatch.setattr(Model, "_initialize", mock_initialize)
        realtime_score_context.prepare_data(".")
        assert realtime_score_context.get_or_create_model()

        assert os.path.isfile("model.pkl")

    def test_mock_get_compute_targets(
        self, realtime_score_context: RealtimeScoreAKSContext
    ):
        """

        :param realtime_score_context: Testing Context
        """
        assert realtime_score_context.compute_targets

    def test_mock_get_webservices(
        self, realtime_score_context: RealtimeScoreAKSContext
    ):
        """

        :param realtime_score_context: Testing Context
        """
        assert realtime_score_context.webservices

    @property
    def models(self):
        """Return a dictionary where the key is model name, and value is a :class:`azureml.core.model.Model` object.

        Raises a :class:`azureml.exceptions.WebserviceException` if there was a problem interacting with
        model management service.

        :return: A dictionary of models.
        :rtype: dict[str, azureml.core.Model]
        :raises: azureml.exceptions.WebserviceException
        """
        return {}


class TestMockDeployRTS(MockWorkspaceCreationTests):
    @pytest.fixture(scope="class")
    def context_type(self):
        """

        :return:
        """
        return MLRealtimeScore

    @pytest.fixture(scope="class")
    def files_for_testing(self):
        return {"train_py": "create_model.py", "score_py": "driver.py"}

    @pytest.fixture
    def realtime_score_context(
        self, monkeypatch, context_type: MLRealtimeScore, files_for_testing
    ) -> MLRealtimeScore:
        """
        Get or Create Context for Testing
        :param files_for_testing:
        :param context_type: impl of WorkspaceContext
        :return:
        """

        def mockreturn(train_py, score_py):
            project_configuration = ProjectConfiguration(project_configuration_file)
            assert project_configuration.has_value("subscription_id")
            assert project_configuration.has_value("resource_group")
            assert project_configuration.has_value("workspace_name")
            ws = MockMLRealtimeScore(
                subscription_id=project_configuration.get_value("subscription_id"),
                resource_group=project_configuration.get_value("resource_group"),
                workspace_name=project_configuration.get_value("workspace_name"),
                configuration_file=project_configuration_file,
                score_py=score_py,
                train_py=train_py,
            )
            return ws

        monkeypatch.setattr(context_type, "get_or_create_workspace", mockreturn)

        return context_type.get_or_create_workspace(
            train_py=files_for_testing["train_py"],
            score_py=files_for_testing["score_py"],
        )


class TestMockDeployDeepRTS(MockWorkspaceCreationTests):
    @pytest.fixture(scope="class")
    def context_type(self):
        """

        :return:
        """
        return DeepRealtimeScore

    @pytest.fixture(scope="class")
    def files_for_testing(self):
        return {"train_py": "create_deep_model.py", "score_py": "deep_driver.py"}

    @pytest.fixture
    def realtime_score_context(
        self, monkeypatch, context_type: MLRealtimeScore, files_for_testing
    ) -> DeepRealtimeScore:
        """
        Get or Create Context for Testing
        :param files_for_testing:
        :param context_type: impl of WorkspaceContext
        :return:
        """

        def mockreturn(train_py, score_py):
            project_configuration = ProjectConfiguration(project_configuration_file)
            assert project_configuration.has_value("subscription_id")
            assert project_configuration.has_value("resource_group")
            assert project_configuration.has_value("workspace_name")
            ws = MockDeepRealtimeScore(
                project_configuration.get_value("subscription_id"),
                project_configuration.get_value("resource_group"),
                project_configuration.get_value("workspace_name"),
                project_configuration_file,
                score_py=score_py,
                train_py=train_py,
            )
            return ws

        monkeypatch.setattr(context_type, "get_or_create_workspace", mockreturn)

        return context_type.get_or_create_workspace(
            train_py=files_for_testing["train_py"],
            score_py=files_for_testing["score_py"],
        )

    def test_mock_get_or_create_model(
        self, monkeypatch, realtime_score_context: DeepRealtimeScore
    ):
        """

        :param realtime_score_context: Testing Context
        """

        @staticmethod
        def mockreturn_2(
            workspace, name, id, tags, properties, version, model_framework, run_id
        ):
            return {
                "name": "mock",
                "id": "1",
                "createdTime": "11/8/2020",
                "description": "",
                "mimeType": "a",
                "properties": "",
                "unpack": "",
                "url": "localhost",
                "version": 1,
                "experimentName": "expName",
                "runId": 1,
                "datasets": None,
                "createdBy": "mock",
                "framework": "python",
                "frameworkVersion": "1",
            }

        def mock_get_model_path_remote(model_name, version, workspace):
            return "."

        def mock_initialize(self, workspace, obj_dict):
            pass

        monkeypatch.setattr(Model, "_get", mockreturn_2)
        monkeypatch.setattr(Model, "_get_model_path_remote", mock_get_model_path_remote)
        monkeypatch.setattr(Model, "_initialize", mock_initialize)
        assert realtime_score_context.get_or_create_model()

        assert os.path.isfile("outputs/model.pkl")

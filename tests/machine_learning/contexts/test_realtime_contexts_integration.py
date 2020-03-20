"""
AI-Utilities - test_realtime_contexts

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import random
import string

import pytest
from azure.mgmt.resource import ResourceManagementClient

from azure_utils.configuration.notebook_config import project_configuration_file
from azure_utils.configuration.project_configuration import ProjectConfiguration
from azure_utils.machine_learning.contexts.model_management_context import (ModelManagementContext, )
from azure_utils.machine_learning.contexts.realtime_score_context import (DeepRealtimeScore, MLRealtimeScore,
                                                                          RealtimeScoreAKSContext, )
from azure_utils.machine_learning.contexts.workspace_contexts import WorkspaceContext


# noinspection PyMethodMayBeStatic
@pytest.mark.skip
class WorkspaceIntegrationTests:
    """Workspace Creation Test Suite"""

    @pytest.fixture(scope="class")
    def unique_configuration(self):
        project_configuration = ProjectConfiguration(project_configuration_file)

        allchar = string.ascii_letters + string.digits
        append = "".join(random.choice(allchar) for _ in range(1, 5))

        settings = [
            "resource_group",
            "workspace_name",
            "image_name",
            "aks_service_name",
            "aks_location",
            "aks_name",
            "deep_image_name",
            "deep_aks_service_name",
            "deep_aks_name",
            "deep_aks_location",
        ]
        for setting in settings:
            project_configuration.append_value(setting, append)
        yield project_configuration
        ws = WorkspaceContext.get_or_create_workspace(
            project_configuration=project_configuration
        )
        rg_client = ResourceManagementClient(
            ws._auth, project_configuration.get_value("subscription_id")
        )
        rg_client.resource_groups.delete(
            resource_group_name=project_configuration.get_value("resource_group")
        )

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

    def test_integration_get_or_create(
        self,
        realtime_score_context: RealtimeScoreAKSContext,
        context_type: WorkspaceContext,
    ):
        """
        Assert Context Type and Creation

        :param realtime_score_context: Testing Context
        :param context_type: Expected Context Type
        """
        assert type(realtime_score_context) is context_type
        assert realtime_score_context

    def test_integration_get_or_create_model(
        self, realtime_score_context: ModelManagementContext
    ):
        """

        :param realtime_score_context: Testing Context
        """
        assert realtime_score_context.get_or_create_model()

    def test_integration_get_or_create_aks(
        self, realtime_score_context: RealtimeScoreAKSContext
    ):
        """

        :param realtime_score_context: Testing Context
        """
        assert realtime_score_context.get_or_create_aks()


class TestIntegrationRTS(WorkspaceIntegrationTests):
    @pytest.fixture(scope="class")
    def realtime_score_context(
        self, context_type: MLRealtimeScore, files_for_testing, unique_configuration
    ) -> MLRealtimeScore:
        """
        Get or Create Context for Testing
        :param files_for_testing:
        :param context_type: impl of WorkspaceContext
        :return:
        """
        return context_type.get_or_create_workspace(
            project_configuration=unique_configuration,
            train_py=files_for_testing["train_py"],
            score_py=files_for_testing["score_py"],
        )

    @pytest.fixture(scope="class")
    def context_type(self):
        """

        :return:
        """
        return MLRealtimeScore

    @pytest.fixture(scope="class")
    def files_for_testing(self):
        return {"train_py": "create_model.py", "score_py": "driver.py"}


class TestDeployDeepRTS(WorkspaceIntegrationTests):
    @pytest.fixture(scope="class")
    def context_type(self):
        """

        :return:
        """
        return DeepRealtimeScore

    @pytest.fixture(scope="class")
    def files_for_testing(self):
        return {"train_py": "create_deep_model.py", "score_py": "score_dl.py"}

    @pytest.fixture(scope="class")
    def realtime_score_context(
        self, context_type: DeepRealtimeScore, files_for_testing, unique_configuration
    ) -> DeepRealtimeScore:
        """
        Get or Create Context for Testing
        :param context_type: impl of WorkspaceContext
        :param test_files: Dict of input Files
        :return:
        """
        return context_type.get_or_create_workspace(
            project_configuration=unique_configuration,
            train_py=files_for_testing["train_py"],
            score_py=files_for_testing["score_py"],
        )

    def test_get_or_create_model(self, realtime_score_context: ModelManagementContext):
        """

        :param realtime_score_context: Testing Context
        """
        assert realtime_score_context.get_or_create_model()

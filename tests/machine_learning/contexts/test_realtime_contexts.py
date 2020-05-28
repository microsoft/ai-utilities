"""
AI-Utilities - test_realtime_contexts

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import os
import os.path
import pytest

from azure_utils.machine_learning.contexts.model_management_context import (
    ModelManagementContext,
)
from azure_utils.machine_learning.contexts.realtime_score_context import (
    DeepRealtimeScore,
    MLRealtimeScore,
    RealtimeScoreAKSContext,
    RealtimeScoreFunctionsContext,
)
from azure_utils.machine_learning.contexts.workspace_contexts import WorkspaceContext


# noinspection PyMethodMayBeStatic
@pytest.mark.smoke
class WorkspaceCreationTests:
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

    def test_get_or_create(
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

    def test_get_or_create_model(self, realtime_score_context: ModelManagementContext):
        """

        :param realtime_score_context: Testing Context
        """
        assert realtime_score_context.get_or_create_model()

    def test_get_images(self, realtime_score_context: RealtimeScoreAKSContext):
        """
        Assert images have been created

        :param realtime_score_context: Testing Context
        """
        assert hasattr(realtime_score_context, "images")

    def test_get_compute_targets(self, realtime_score_context: RealtimeScoreAKSContext):
        """

        :param realtime_score_context: Testing Context
        """
        assert hasattr(realtime_score_context, "compute_targets")

    def test_get_webservices(self, realtime_score_context: RealtimeScoreAKSContext):
        """

        :param realtime_score_context: Testing Context
        """
        assert hasattr(realtime_score_context, "webservices")

    def test_get_or_create_aks(self, realtime_score_context: RealtimeScoreAKSContext):
        """

        :param realtime_score_context: Testing Context
        """
        assert realtime_score_context.get_or_create_aks()


class TestDeployRTS(WorkspaceCreationTests):
    @pytest.fixture(scope="class")
    def realtime_score_context(
        self, context_type: MLRealtimeScore, files_for_testing
    ) -> MLRealtimeScore:
        """
        Get or Create Context for Testing
        :param files_for_testing:
        :param context_type: impl of WorkspaceContext
        :return:
        """
        return context_type.get_or_create_workspace(
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


class TestDeployDeepRTS(WorkspaceCreationTests):
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
        self, context_type: DeepRealtimeScore, files_for_testing
    ) -> DeepRealtimeScore:
        """
        Get or Create Context for Testing
        :param context_type: impl of WorkspaceContext
        :param test_files: Dict of input Files
        :return:
        """
        return context_type.get_or_create_workspace(
            train_py=files_for_testing["train_py"],
            score_py=files_for_testing["score_py"],
        )

    def test_get_or_create_model(self, realtime_score_context: ModelManagementContext):
        """

        :param realtime_score_context: Testing Context
        """
        assert realtime_score_context.get_or_create_model()

    def test_get_or_create_webservices(self, realtime_score_context: DeepRealtimeScore, files_for_testing):
        """

        :param realtime_score_context: Testing Context
        """

        if not os.path.isfile(f"source/{files_for_testing['score_py']}"):
            os.makedirs("source", exist_ok=True)

            score_py = """
            import sys
            sys.setrecursionlimit(3000)

            from azureml.contrib.services.aml_request import rawhttp

            def init():
                global process_and_score
                from azure_utils.samples.deep_rts_samples import get_model_api
                process_and_score = get_model_api()


            @rawhttp
            def run(request):
                from azure_utils.machine_learning.realtime import default_response
                if request.method == 'POST':
                    return process_and_score(request.files)
                return default_response(request)

            """
            with open(f"source/{files_for_testing['score_py']}", "w") as file:
                file.write(score_py)

        model = realtime_score_context.get_or_create_model()
        inference_config = realtime_score_context.get_inference_config()
        aks_target = realtime_score_context.get_or_create_aks()
        web_service = realtime_score_context.get_or_create_aks_service(model, aks_target, inference_config)
        assert web_service.state == "Healthy"

    def test_web_service(self, realtime_score_context: DeepRealtimeScore):
        realtime_score_context.test_service_local()


# noinspection PyUnresolvedReferences,PyUnresolvedReferences,PyUnresolvedReferences
class TestDeployDeepRTSLocally:
    def dont_test_train_py(self):
        if not os.path.isdir("outputs"):
            os.mkdir("../outputs")
        if os.path.isfile("script/create_deep_model_new.py"):
            os.system("python script/create_deep_model_new.py")

            assert os.path.isfile("../outputs/model.pkl")


def dont_test_get_or_create_function_endpoint():
    """Test creation of Azure Function for ML Scoring"""
    RealtimeScoreFunctionsContext.get_or_or_create_function_endpoint()

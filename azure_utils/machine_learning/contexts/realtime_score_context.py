"""
AI-Utilities - realtime_score_context.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import json
import os
import time
from abc import ABC
from typing import Any, Tuple

from azure.mgmt.deploymentmanager.models import DeploymentMode
from azure.mgmt.resource import ResourceManagementClient
from azureml.accel import AccelContainerImage, AccelOnnxConverter, PredictionClient
from azureml.accel.models import QuantizedResnet50, utils as utils
from azureml.contrib.functions import HTTP_TRIGGER, package
from azureml.core import ComputeTarget, Environment, Image, Model, Webservice, Workspace
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.compute import AksCompute
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.image import ContainerImage
from azureml.core.image.container import ContainerImageConfig
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AksWebservice
from azureml.core.webservice.aks import AksServiceDeploymentConfiguration
from azureml.exceptions import WebserviceException
from deprecated import deprecated

from azure_utils import directory
from azure_utils.configuration.notebook_config import (
    project_configuration_file,
    score_py_default,
    train_py_default,
)
from azure_utils.configuration.project_configuration import ProjectConfiguration
from azure_utils.machine_learning.contexts.model_management_context import (
    LocalTrainingContext,
)
from azure_utils.machine_learning.contexts.workspace_contexts import WorkspaceContext
from azure_utils.machine_learning.datasets.stack_overflow_data import (
    clean_data,
    download_datasets,
    save_data,
    split_duplicates,
)
from azure_utils.machine_learning.deep.create_deep_model import (
    get_or_create_resnet_image,
)
from azure_utils.machine_learning.realtime.image import (
    get_or_create_lightgbm_image,
    print_deployment_time,
    print_image_deployment_info,
)
from azure_utils.notebook_widgets.workspace_widget import make_workspace_widget
import azureml.accel.models.utils as utils


class RealtimeScoreContext(WorkspaceContext):
    """ Real-time Score Context """

    def __init__(
        self,
        subscription_id,
        resource_group,
        workspace_name,
        configuration_file: str = project_configuration_file,
        project_configuration=None,
        score_py=score_py_default,
        settings_image_name="image_name",
        settings_aks_name="aks_name",
        settings_aks_service_name="aks_service_name",
        wait_for_completion=True,
        **kwargs,
    ):
        if not project_configuration:
            project_configuration = ProjectConfiguration(configuration_file)
        super().__init__(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            **kwargs,
        )
        self.project_configuration = project_configuration

        self.score_py = score_py

        self.settings_image_name = settings_image_name
        self.settings_aks_name = settings_aks_name
        self.settings_aks_service_name = settings_aks_service_name

        # Model Configuration
        self.model_name = "default_model_name"
        self.model_tags = None
        self.model_description = None

        # Conda Configuration
        self.conda_file = None
        self.get_details()
        self.conda_pack = None
        self.requirements = None
        self.conda_env = CondaDependencies.create(
            conda_packages=self.conda_pack, pip_packages=self.requirements
        )

        # Env Configuration
        self.env_dependencies = None

        self.dockerfile = None
        self.image_dependencies = None
        self.image_description = None
        self.image_enable_gpu = False

        self.get_image = None
        self.aks_vm_size: str = "Standard_D2_v2"

        self.num_estimators = "1"

        self.node_count = 6
        self.num_replicas: int = 2
        self.cpu_cores: int = 1

        self.workspace_widget = None
        self.wait_for_completion = wait_for_completion

    def test_service_local(self) -> None:
        """
        Test Scoring Service Locally by loading file
        """

        exec(open("source/"+ self.score_py).read())
        exec("init()")
        exec("response = run(MockRequest())")
        exec("assert response")
        exec("response = run(MockImageRequest())")
        exec("assert response")

    def get_inference_config(self) -> InferenceConfig:
        """
        Get Inference Configuration
        :return:
        """
        environment = Environment("conda-env")
        environment.python.conda_dependencies = self.conda_env

        inference_config = InferenceConfig(
            entry_script=self.score_py,
            environment=environment,
            source_directory="source",
        )
        return inference_config

    def write_conda_env(self) -> None:
        """
        Write Conda Config to file.
        """
        with open(self.conda_file, "w") as file:
            file.write(self.conda_env.serialize_to_string())

    def assert_image_params(self) -> None:
        """
        Assert required params
        """
        assert self.score_py
        assert self.conda_file
        assert self.image_description
        assert self.image_tags


class RealtimeScoreFunctionsContext(RealtimeScoreContext, LocalTrainingContext, ABC):
    """ Realtime Scoring Function Context"""

    @classmethod
    def get_or_or_create_function_endpoint(cls) -> Tuple[Workspace, Image]:
        """ Get or Create Real-time Endpoint """
        workspace = cls.get_or_create_workspace()
        model = workspace.get_or_create_model()
        config = workspace.get_or_create_function_image_configuration()
        image = workspace.get_or_create_function_image(config, models=[model])
        return workspace, image

    def get_or_create_function_image_configuration(self) -> InferenceConfig:
        """ Get or Create new Docker Image Configuration for Machine Learning Workspace

        Image Configuration for running LightGBM in Azure Machine Learning Workspace

        :return: new image configuration for Machine Learning Workspace
        """
        self.assert_image_params()

        from azureml.core.environment import Environment
        from azureml.core.conda_dependencies import CondaDependencies

        # Create an environment and add conda dependencies to it
        myenv = Environment(name="myenv")
        # Enable Docker based environment
        myenv.docker.enabled = True
        # Build conda dependencies
        myenv.python.conda_dependencies = CondaDependencies.create(
            conda_packages=["scikit-learn"], pip_packages=["azureml-defaults"]
        )
        return InferenceConfig(entry_script=self.score_py, environment=myenv)

    def get_or_create_function_image(self, config: InferenceConfig, models: list):
        """
        Create new configuration for deploying an scoring service to function image.

        :param config:
        :param models:
        """
        blob = package(
            self, models, config, functions_enabled=True, trigger=HTTP_TRIGGER
        )
        blob.wait_for_creation(show_output=True)
        # Display the package location/ACR path
        print(blob.location)


class RealtimeScoreAKSContext(RealtimeScoreContext):
    """
    Real-time Scoring with AKS Interface
    """

    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        workspace_name: str,
        configuration_file: str = project_configuration_file,
        project_configuration=None,
        score_py=score_py_default,
        settings_aks_name="aks_name",
        settings_aks_service_name="aks_service_name",
        **kwargs,
    ):
        """
        Create new Real-time Scoring on AKS Context.

        :param subscription_id: Azure subscription id
        :param resource_group: Azure Resource Group name
        :param workspace_name: Azure Machine Learning Workspace
        :param kwargs: additional args
        """
        if not project_configuration:
            project_configuration = ProjectConfiguration(configuration_file)
        super().__init__(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            project_configuration=project_configuration,
            score_py=score_py,
            **kwargs,
        )
        self.aks_name = self.assert_and_get_value(settings_aks_name)
        self.aks_service_name = self.assert_and_get_value(settings_aks_service_name)
        self.aks_location = self.assert_and_get_value("workspace_region")

    @classmethod
    def get_or_or_create(
        cls,
        configuration_file: str = project_configuration_file,
        train_py=train_py_default,
        score_py=score_py_default,
    ):
        """ Get or Create Real-time Endpoint on AKS

        :param configuration_file: path to project configuration file. default: project.yml
        :param train_py: python source file for training
        :param score_py: python source file for scoring
        """
        print("Step 1: Get or Create Model")
        model, workspace = cls._get_workspace_and_model(
            configuration_file, score_py, train_py
        )
        print("Step 2: Create Inference Configuration")
        inference_config = workspace.get_inference_config()
        print("Step 3: Get or Create Kubernetes Cluster")
        aks_target = workspace.get_or_create_aks()
        print("Step 4: Get or Create Web Service")
        web_service = workspace.get_or_create_aks_service(
            model, aks_target, inference_config
        )
        print("All Steps Completed")
        return workspace, web_service

    @classmethod
    def _get_workspace_and_model(
        cls, configuration_file: str, score_py: str, train_py: str
    ) -> Tuple[Model, Any]:
        """
        Retrieve Workspace and Model

        :param configuration_file: path to project configuration file
        :param score_py: python source file for scoring
        :param train_py: python source file for training
        :return: Model and Workspace
        """
        workspace = cls.get_or_create_workspace(
            configuration_file, train_py=train_py, score_py=score_py
        )
        model = workspace.get_or_create_model()
        return model, workspace

    def get_or_create_aks(self) -> ComputeTarget:
        """
        Get or Create AKS Cluster

        :return: AKS Compute from Workspace
        """
        assert "_" not in self.project_configuration.get_value(
            self.settings_aks_service_name
        ), (self.settings_aks_service_name + " can not contain _")
        assert self.project_configuration.has_value(
            "workspace_region"
        ), """Configuration does not have a Workspace Region"""

        workspace_compute = self.compute_targets
        print("Check if Cluster exists.")
        if self.aks_name in workspace_compute:
            print("Cluster does exists.")
            return AksCompute(self, self.aks_name)
        print("Cluster does not exists.")
        prov_config = AksCompute.provisioning_configuration(
            agent_count=self.node_count,
            vm_size=self.aks_vm_size,
            location=self.aks_location,
        )

        deploy_aks_start = time.time()
        aks_target = ComputeTarget.create(
            workspace=self, name=self.aks_name, provisioning_configuration=prov_config
        )

        aks_target.wait_for_completion(show_output=True)
        if self.show_output:
            service_name = "AKS"
            print_deployment_time(self.aks_service_name, deploy_aks_start, service_name)
            print(aks_target.provisioning_state)
            print(aks_target.provisioning_errors)
        aks_status = aks_target.get_status()
        assert aks_status == "Succeeded", "AKS failed to create"
        return aks_target

    def get_aks_deployment_config(self) -> AksServiceDeploymentConfiguration:
        """

        :return:
        """
        aks_deployment_configuration = {
            "num_replicas": self.num_replicas,
            "cpu_cores": self.cpu_cores,
            "enable_app_insights": True,
            "collect_model_data": True,
        }
        return AksWebservice.deploy_configuration(**aks_deployment_configuration)

    def get_or_create_aks_service(
        self, model: Model, aks_target: AksCompute, inference_config: InferenceConfig
    ) -> Webservice:
        """

        :param model:
        :param aks_target:
        :param inference_config:
        :return:
        """
        model_dict = model.serialize()

        print("Check if AKS Service Exists")
        if self.aks_service_name in self.webservices:
            print("AKS Service Exists")
            aks_service = AksWebservice(self, self.aks_service_name)
            if aks_service.state == "Succeeded":
                self._post_process_aks_deployment(aks_service, aks_target, model_dict)
                return aks_service
        print("AKS Service Does Not Exists")
        print("Test Score File Locally - Begin")
        # test_score_file("source/score.py")
        print("Test Score File Locally - Success")
        print("Model Deploy - Begin")
        aks_service = Model.deploy(
            self,
            self.aks_service_name,
            models=[model],
            inference_config=inference_config,
            deployment_target=aks_target,
            overwrite=True,
        )
        self._post_process_aks_deployment(aks_service, aks_target, model_dict)
        try:
            if self.wait_for_completion:
                self.wait_then_configure_ping_test(aks_service, self.aks_service_name)
                print("Model Deploy - Success")
        finally:
            if self.show_output:
                print(aks_service.get_logs())
        return aks_service

    def _post_process_aks_deployment(
        self, aks_service: AksWebservice, aks_target: AksCompute, model_dict: dict
    ):
        aks_dict = aks_service.serialize()
        self.workspace_widget = make_workspace_widget(model_dict, aks_dict)
        # self.create_kube_config(aks_target)

    def wait_then_configure_ping_test(
        self, aks_service: AksWebservice, aks_service_name
    ):
        """

        :param aks_service:
        :param aks_service_name:
        """
        try:
            aks_service.wait_for_deployment(show_output=self.show_output)
        except WebserviceException:
            print(aks_service.get_logs())
            raise Exception(aks_service.get_logs())
        # self.configure_ping_test(
        #     "ping-test-" + aks_service_name,
        #     self.get_details()["applicationInsights"],
        #     aks_service.scoring_uri,
        #     aks_service.get_keys()[0],
        # )

    def has_web_service(self, service_name: str) -> bool:
        """

        :param service_name:
        :return:
        """
        return service_name in self.webservices

    def get_web_service_state(self, service_name: str) -> str:
        """

        :param service_name:
        :return:
        """
        web_service = self.get_web_service(service_name)
        web_service.update_deployment_state()
        assert web_service.state
        return web_service.state

    def get_web_service(self, service_name: str) -> Webservice:
        """

        :param service_name:
        :return:
        """
        assert self.webservices[service_name]
        return Webservice(self, service_name)

    @staticmethod
    def create_kube_config(aks_target: AksCompute):
        """

        :param aks_target:
        """
        user_path = os.path.expanduser("~")
        kub_dir = os.path.join(user_path, ".kube")
        config_path = os.path.join(kub_dir, "config")

        os.makedirs(kub_dir, exist_ok=True)
        with open(config_path, "a") as f:
            f.write(aks_target.get_credentials()["userKubeConfig"])

    def _aks_exists(self) -> bool:
        """Check if Kubernetes Cluster Exists or has Failed"""
        if (
            self.aks_name in self.compute_targets
            and AksCompute(self, self.aks_name).provisioning_state != "Failed"
        ):
            return True
        return False

    @staticmethod
    def configure_ping_test(
        ping_test_name: str, app_name: str, ping_url: str, ping_token: str
    ):
        """

        :param ping_test_name:
        :param app_name:
        :param ping_url:
        :param ping_token:
        """
        project_configuration = ProjectConfiguration(project_configuration_file)
        assert project_configuration.has_value("subscription_id")
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message=r"track*")
        credentials = AzureCliAuthentication()
        client = ResourceManagementClient(
            credentials, project_configuration.get_value("subscription_id")
        )
        template_path = os.path.join(
            os.path.dirname(__file__), "templates", "webtest.json"
        )
        with open(template_path) as template_file_fd:
            template = json.load(template_file_fd)

        parameters = {
            "appName": app_name.split("components/")[1],
            "pingURL": ping_url,
            "pingToken": ping_token,
            "location": project_configuration.get_value("workspace_region"),
            "pingTestName": ping_test_name
            + "-"
            + project_configuration.get_value("workspace_region"),
        }
        parameters = {k: {"value": v} for k, v in parameters.items()}

        deployment_properties = {
            "mode": DeploymentMode.incremental,
            "template": template,
            "parameters": parameters,
        }

        deployment_async_operation = client.deployments.create_or_update(
            project_configuration.get_value("resource_group"),
            "add-web-test",
            deployment_properties,
        )
        deployment_async_operation.wait()


@deprecated(
    version="0.3.81", reason="Switch to using Env, this will be removed in 0.4.0"
)
class RealTimeScoreImageAndAKSContext(RealtimeScoreAKSContext):
    """ Old way to deploy AKS with Docker Image, use RealtimeScoreAKSContext instead"""

    @deprecated(
        version="0.3.81", reason="Switch to using Env, this will be removed in 0.4.0"
    )
    def get_or_create_aks(self, aks_target) -> AksWebservice:
        """
        Get or Create AKS Service with new or existing Kubernetes Compute

        :param aks_target:
        :return: New or Existing Kubernetes Web Service
        """
        image_name = self.assert_and_get_value(self.settings_image_name)
        assert self.aks_vm_size

        if self._aks_exists():
            # self.create_kube_config(aks_target)
            return self.get_web_service(self.aks_service_name)

        aks_config = self.get_aks_deployment_config()

        if image_name not in self.images:
            self.get_image()
        image = self.images[image_name]

        deploy_from_image_start = time.time()

        aks_service = Webservice.deploy_from_image(
            workspace=self,
            name=self.aks_service_name,
            image=image,
            deployment_config=aks_config,
            deployment_target=aks_target,
            overwrite=True,
        )
        try:
            self.wait_then_configure_ping_test(aks_service, self.aks_service_name)
        except WebserviceException:
            print(aks_service.get_logs())
            raise
        if self.show_output:
            print_deployment_time(self.aks_service_name, deploy_from_image_start, "AKS")
            print(aks_service.state)
            print(aks_service.get_logs())

        return aks_service

    @deprecated(
        version="0.3.81", reason="Switch to using Env, this will be removed in 0.4.0"
    )
    def get_inference_config(self, kwargs: dict) -> ContainerImageConfig:
        """ Get or Create new Docker Image Configuration for Machine Learning Workspace

        Image Configuration for running LightGBM in Azure Machine Learning Workspace

        :param kwargs: keyword args
        :return: new image configuration for Machine Learning Workspace
        """
        self.assert_image_params()

        self.write_conda_env()
        assert os.path.isfile(self.conda_file)

        self.image_tags["score_py_hash"] = self._get_file_md5(self.score_py)
        return ContainerImage.image_configuration(
            execution_script=self.score_py,
            runtime="python",
            conda_file=self.conda_file,
            description=self.image_description,
            dependencies=self.image_dependencies,
            docker_file=self.dockerfile,
            tags=self.image_tags,
            enable_gpu=self.image_enable_gpu,
            **kwargs,
        )

    @deprecated(
        version="0.3.81", reason="Switch to using Env, this will be removed in 0.4.0"
    )
    def get_or_create_image(
        self, image_config: ContainerImageConfig, models: list = None
    ) -> Image:
        """Get or Create new Docker Image from Machine Learning Workspace

        :param image_config:
        :param models:
        """
        if not models:
            models = []

        image_name = self.assert_and_get_value(self.settings_image_name)

        if (
            image_name in self.images
            and self.images[image_name].creation_state != "Failed"
        ):
            # hasher = hashlib.md5()
            # with open(self.score_py, 'rb') as afile:
            #     buf = afile.read()
            #     hasher.update(buf)
            # if "hash" in Image(workspace, image_name).tags \
            #         and hasher.hexdigest() == Image(workspace, image_name).tags['hash']:
            return self.images[image_name]

        image_create_start = time.time()
        image = ContainerImage.create(
            name=image_name, models=models, image_config=image_config, workspace=self
        )
        image.wait_for_creation(show_output=self.show_output)
        assert image.creation_state != "Failed"
        if self.show_output:
            print_image_deployment_info(image, image_name, image_create_start)
        return image


class MLRealtimeScore(RealtimeScoreAKSContext, LocalTrainingContext):
    """ Light GBM Real Time Scoring"""

    def __init__(
        self,
        subscription_id,
        resource_group,
        workspace_name,
        configuration_file: str = project_configuration_file,
        project_configuration=None,
        train_py=train_py_default,
        score_py=score_py_default,
        conda_file="my_env.yml",
        **kwargs,
    ):
        if not project_configuration:
            project_configuration = ProjectConfiguration(configuration_file)
        super().__init__(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            project_configuration=project_configuration,
            score_py=score_py,
            **kwargs,
        )
        self.score_py = score_py
        self.train_py = train_py
        self.get_docker_file()
        self.setting_image_name = "image_name"
        self.settings_aks_name = "aks_name"
        self.settings_aks_service_name = "aks_service_name"

        self.execution_script = "score.py"
        with open(self.execution_script, "w") as file:
            file.write(
                """
import json
import os
import logging

from flask import Flask

Flask(__name__).config['OPENCENSUS'] = {
    'TRACE': {
        'SAMPLER': 'opencensus.trace.samplers.ProbabilitySampler(rate=1.0)',
        'EXPORTER': '''opencensus.ext.azure.trace_exporter.AzureExporter(
            connection_string="InstrumentationKey=" + ''' + os.getenv('AML_APP_INSIGHTS_KEY') + ''',
        )'''
    }
}


def init():
    logger = logging.getLogger("scoring_script")
    logger.info("init")


def run():
    logger = logging.getLogger("scoring_script")
    logger.info("run")
    return json.dumps({'call': True})

"""
            )

        self.model_name = "question_match_model"
        self.experiment_name = "mlaks-train-on-local"
        self.model_path = "./outputs/model.pkl"

        self.source_directory = "./script"
        self.script = "create_model.py"
        self.create_model_script_file()

        self.show_output = True
        self.args = [
            "--inputs",
            os.path.abspath(directory + "/data_folder"),
            "--outputs",
            "outputs",
            "--estimators",
            self.num_estimators,
            "--match",
            "2",
        ]

        self.test_size = 0.21
        self.min_text = 150
        self.min_dupes = 12
        self.match = 20

        self.prepare_data()
        # Image Configuration
        self.get_image = get_or_create_lightgbm_image
        self.image_tags = {
            "area": "text",
            "type": "lightgbm",
            "name": "AKS",
            "project": "AML",
        }
        self.image_description = "Image for lightgbm model"
        self.image_dependencies = None
        self.image_enable_gpu = False

        self.conda_file = conda_file
        self.conda_pack = ["scikit-learn==0.19.1", "pandas==0.23.3"]
        self.requirements = [
            "lightgbm==2.1.2",
            "azureml-defaults==1.0.57",
            "azureml-contrib-services",
            "opencensus-ext-flask",
            "Microsoft-AI-Azure-Utility-Samples",
        ]
        self.conda_env = CondaDependencies.create(
            conda_packages=self.conda_pack, pip_packages=self.requirements
        )

    def create_model_script_file(self) -> None:
        """ Adding Model Script"""
        if not os.path.isfile(self.source_directory + "/" + self.script):
            os.makedirs("script", exist_ok=True)

            create_model_py = """
import argparse
import os

import lightgbm as lgb
import pandas as pd
from azureml.core import Run
import joblib
from sklearn.feature_extraction import text
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline

from azure_utils.machine_learning.item_selector import ItemSelector

if __name__ == '__main__':
    # Define the arguments.
    parser = argparse.ArgumentParser(description='Fit and evaluate a model based on train-test datasets.')
    parser.add_argument('-d', '--train_data', help='the training dataset name', default='balanced_pairs_train.tsv')
    parser.add_argument('-t', '--test_data', help='the test dataset name', default='balanced_pairs_test.tsv')
    parser.add_argument('-i', '--estimators', help='the number of learner estimators', type=int, default=1)
    parser.add_argument('--min_child_samples', help='the minimum number of samples in a child(leaf)', type=int,
                        default=20)
    parser.add_argument('-v', '--verbose', help='the verbosity of the estimator', type=int, default=-1)
    parser.add_argument('-n', '--ngrams', help='the maximum size of word ngrams', type=int, default=1)
    parser.add_argument('-u', '--unweighted', help='do not use instance weights', action='store_true', default=False)
    parser.add_argument('-m', '--match', help='the maximum number of duplicate matches', type=int, default=20)
    parser.add_argument('--outputs', help='the outputs directory', default='.')
    parser.add_argument('--inputs', help='the inputs directory', default='.')
    parser.add_argument('-s', '--save', help='save the model', action='store_true', default=True)
    parser.add_argument('--model', help='the model file', default='model.pkl')
    parser.add_argument('--instances', help='the instances file', default='inst.txt')
    parser.add_argument('--labels', help='the labels file', default='labels.txt')
    parser.add_argument('-r', '--rank', help='the maximum rank of correct answers', type=int, default=3)
    args = parser.parse_args()

    run = Run.get_context()

    # The training and testing datasets.
    inputs_path = args.inputs
    data_path = os.path.join(inputs_path, args.train_data)
    test_path = os.path.join(inputs_path, args.test_data)

    # Create the outputs folder.
    outputs_path = args.outputs
    os.makedirs(outputs_path, exist_ok=True)
    model_path = os.path.join(outputs_path, args.model)
    instances_path = os.path.join(outputs_path, args.instances)
    labels_path = os.path.join(outputs_path, args.labels)

    # Load the training data
    assert os.path.isfile(data_path)
    print('Reading {}'.format(data_path))
    train = pd.read_csv(data_path, sep='\t', encoding='latin1')

    # Limit the number of duplicate-original question matches.
    train = train[train.n < args.match]

    # Define the roles of the columns in the training data.
    feature_columns = ['Text_x', 'Text_y']
    label_column = 'Label'
    duplicates_id_column = 'Id_x'
    answer_id_column = 'AnswerId_y'

    # Report on the training dataset: the number of rows and the proportion of true matches.
    print('train: {:,} rows with {:.2%} matches'.format(
        train.shape[0], train[label_column].mean()))

    # Compute the instance weights used to correct for class imbalance in training.
    weight_column = 'Weight'
    if args.unweighted:
        weight = pd.Series([1.0], train[label_column].unique())
    else:
        label_counts = train[label_column].value_counts()
        weight = train.shape[0] / (label_counts.shape[0] * label_counts)
    train[weight_column] = train[label_column].apply(lambda x: weight[x])

    # Collect the unique ids that identify each original question's answer.
    labels = sorted(train[answer_id_column].unique())
    label_order = pd.DataFrame({'label': labels})

    # Collect the parts of the training data by role.
    train_x = train[feature_columns]
    train_y = train[label_column]
    sample_weight = train[weight_column]

    # Use the inputs to define the hyperparameters used in training.
    n_estimators = args.estimators
    min_child_samples = args.min_child_samples
    if args.ngrams > 0:
        ngram_range = (1, args.ngrams)
    else:
        ngram_range = None

    # Verify that the hyperparameter values are valid.
    assert n_estimators > 0
    assert min_child_samples > 1
    assert isinstance(ngram_range, tuple) and len(ngram_range) == 2
    assert 0 < ngram_range[0] <= ngram_range[1]

    # Define the pipeline that featurizes the text columns.
    featurization = [
        (column,
         make_pipeline(ItemSelector(column),
                       text.TfidfVectorizer(ngram_range=ngram_range)))
        for column in feature_columns]
    features = FeatureUnion(featurization)

    # Define the estimator that learns how to classify duplicate-original question pairs.
    estimator = lgb.LGBMClassifier(n_estimators=n_estimators,
                                   min_child_samples=min_child_samples,
                                   verbose=args.verbose)

    # Define the model pipeline as feeding the features into the estimator.
    model = Pipeline([
        ('features', features),
        ('model', estimator)
    ])

    # Fit the model.
    print('Training...')
    model.fit(train_x, train_y, model__sample_weight=sample_weight)

    # Save the model to a file, and report on its size.
    if args.save:
        joblib.dump(model, model_path)
        print('{} size: {:.2f} MB'.format(model_path, os.path.getsize(model_path) / (2 ** 20)))

"""
            with open(self.source_directory + "/" + self.script, "w") as file:
                file.write(create_model_py)

    def get_docker_file(self) -> None:
        """
        Create Docker File with GCC
        """
        self.dockerfile = "dockerfile"
        with open(self.dockerfile, "w") as file:
            file.write(
                "RUN apt update -y && apt upgrade -y && apt install -y build-essential"
            )

    def prepare_data(self, outputs_path=directory + "/data_folder") -> None:
        """
        Prepare the training data
        """
        dupes_test_path = os.path.join(outputs_path, "dupes_test.tsv")
        questions_path = os.path.join(outputs_path, "questions.tsv")

        if not (os.path.isfile(dupes_test_path) and os.path.isfile(questions_path)):
            answers, dupes, questions = download_datasets(show_output=self.show_output)

            # Clean up all text, and keep only data with some clean text.
            dupes, label_column, questions = clean_data(
                answers,
                dupes,
                self.min_dupes,
                self.min_text,
                questions,
                self.show_output,
            )

            # Split dupes into train and test ensuring at least one of each label class is in test.
            balanced_pairs_test, balanced_pairs_train, dupes_test = split_duplicates(
                dupes,
                label_column,
                self.match,
                questions,
                self.show_output,
                self.test_size,
            )

            save_data(
                balanced_pairs_test,
                balanced_pairs_train,
                dupes_test,
                dupes_test_path,
                outputs_path,
                questions,
                questions_path,
                self.show_output,
            )


class DeepRealtimeScore(
    RealtimeScoreAKSContext, RealtimeScoreFunctionsContext, LocalTrainingContext
):
    """ Resnet Real-time Scoring"""

    def __init__(
        self,
        subscription_id,
        resource_group,
        workspace_name,
        configuration_file: str = project_configuration_file,
        train_py=train_py_default,
        score_py=score_py_default,
        conda_file="my_env.yml",
        setting_image_name="deep_image_name",
        settings_aks_name="deep_aks_name",
        settings_aks_service_name="deep_aks_service_name",
        **kwargs,
    ):
        super().__init__(
            subscription_id,
            resource_group,
            workspace_name,
            configuration_file=configuration_file,
            train_py=train_py,
            settings_aks_name=settings_aks_name,
            settings_aks_service_name=settings_aks_service_name,
        )
        self.setting_image_name = setting_image_name
        self.settings_aks_name = settings_aks_name
        self.settings_aks_service_name = settings_aks_service_name
        self.score_py = score_py
        self.train_py = train_py
        # Experiment Configuration
        self.experiment_name = "dlrts-train-on-local"

        # Model Configuration
        self.model_name = "resnet_model_2"
        self.model_tags = {"model": "dl", "framework": "resnet"}
        self.model_description = "resnet 152 model"
        self.model_path = "./outputs/model.pkl"
        self.args = [
            "--outputs",
            "outputs",
            "--estimators",
            self.num_estimators,
            "--match",
            "2",
        ]
        # Conda Configuration
        self.conda_file = conda_file
        self.write_conda_env()
        self.conda_pack = ["keras=2.3.1", "pillow=7.0.0", "lightgbm=2.3.1"]
        self.requirements = [
            "azureml-defaults",
            "azureml-contrib-services",
            "toolz==0.10.0",
            "git+https://github.com/microsoft/AI-Utilities.git",
        ]
        self.conda_env = CondaDependencies.create(
            conda_packages=self.conda_pack, pip_packages=self.requirements
        )

        # Env Configuration
        self.env_dependencies = ["resnet152.py"]

        # Image Configuration
        self.get_image = get_or_create_resnet_image
        self.image_tags = {"name": "AKS", "project": "AML"}
        self.image_description = "Image for AKS Deployment Tutorial"
        self.image_dependencies = ["resnet152.py"]
        self.image_enable_gpu = True

        # Kubernetes Configuration
        self.aks_vm_size = "Standard_NC6"
        self.source_directory = "./script"

        if not os.path.isfile(self.source_directory + "/" + self.train_py):
            os.makedirs(self.source_directory, exist_ok=True)

            create_model_py = """

import keras.backend as K
from keras import initializers
from keras.engine import Layer, InputSpec
from keras.engine.topology import get_source_inputs
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import add
from keras.models import Model
from keras.utils import layer_utils
from keras.utils.data_utils import get_file

WEIGHTS_PATH = "https://github.com/adamcasson/resnet152/releases/download/v0.1/resnet152_weights_tf.h5"
WEIGHTS_PATH_NO_TOP = "https://github.com/adamcasson/resnet152/releases/download/v0.1/resnet152_weights_tf_notop.h5"

def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        require_flatten,
                        weights=None):
    if weights != 'imagenet' and input_shape and len(input_shape) == 3:
        if data_format == 'channels_first':
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[0]) + ' input channels.')
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[-1]) + ' input channels.')
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if weights == 'imagenet' and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting`include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be ' +
                                 str(default_shape) + '.')
        return default_shape
    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[0] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[1] is not None and input_shape[1] < min_size) or
                   (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) + '; got '
                                     '`input_shape=' + str(input_shape) + '`')
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[-1] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[0] is not None and input_shape[0] < min_size) or
                   (input_shape[1] is not None and input_shape[1] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) + '; got '
                                     '`input_shape=' + str(input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape


class Scale(Layer):

    def __init__(
        self,
        weights=None,
        axis=-1,
        momentum=0.9,
        beta_init="zero",
        gamma_init="one",
        **kwargs
    ):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(self.gamma_init(shape), name="%s_gamma" % self.name)
        self.beta = K.variable(self.beta_init(shape), name="%s_beta" % self.name)
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(
            self.beta, broadcast_shape
        )
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def identity_block(input_tensor, kernel_size, filters, stage, block):
    eps = 1.1e-5

    if K.common.image_dim_ordering() == "tf":
        bn_axis = 3
    else:
        bn_axis = 1

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"
    scale_name_base = "scale" + str(stage) + block + "_branch"

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + "2a", use_bias=False)(
        input_tensor
    )
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + "2a")(x)
    x = Scale(axis=bn_axis, name=scale_name_base + "2a")(x)
    x = Activation("relu", name=conv_name_base + "2a_relu")(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + "2b_zeropadding")(x)
    x = Conv2D(
        nb_filter2,
        (kernel_size, kernel_size),
        name=conv_name_base + "2b",
        use_bias=False,
    )(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + "2b")(x)
    x = Scale(axis=bn_axis, name=scale_name_base + "2b")(x)
    x = Activation("relu", name=conv_name_base + "2b_relu")(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + "2c", use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + "2c")(x)
    x = Scale(axis=bn_axis, name=scale_name_base + "2c")(x)

    x = add([x, input_tensor], name="res" + str(stage) + block)
    x = Activation("relu", name="res" + str(stage) + block + "_relu")(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    eps = 1.1e-5

    if K.common.image_dim_ordering() == "tf":
        bn_axis = 3
    else:
        bn_axis = 1

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"
    scale_name_base = "scale" + str(stage) + block + "_branch"

    x = Conv2D(
        nb_filter1, (1, 1), strides=strides, name=conv_name_base + "2a", use_bias=False
    )(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + "2a")(x)
    x = Scale(axis=bn_axis, name=scale_name_base + "2a")(x)
    x = Activation("relu", name=conv_name_base + "2a_relu")(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + "2b_zeropadding")(x)
    x = Conv2D(
        nb_filter2,
        (kernel_size, kernel_size),
        name=conv_name_base + "2b",
        use_bias=False,
    )(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + "2b")(x)
    x = Scale(axis=bn_axis, name=scale_name_base + "2b")(x)
    x = Activation("relu", name=conv_name_base + "2b_relu")(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + "2c", use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + "2c")(x)
    x = Scale(axis=bn_axis, name=scale_name_base + "2c")(x)

    shortcut = Conv2D(
        nb_filter3, (1, 1), strides=strides, name=conv_name_base + "1", use_bias=False
    )(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + "1")(
        shortcut
    )
    shortcut = Scale(axis=bn_axis, name=scale_name_base + "1")(shortcut)

    x = add([x, shortcut], name="res" + str(stage) + block)
    x = Activation("relu", name="res" + str(stage) + block + "_relu")(x)
    return x


def ResNet152(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    large_input=False,
    pooling=None,
    classes=1000,
):
    if weights not in {"imagenet", None}:
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization) or `imagenet` "
            "(pre-training on ImageNet)."
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            "If using `weights` as imagenet with `include_top`"
            " as true, `classes` should be 1000"
        )

    eps = 1.1e-5

    if large_input:
        img_size = 448
    else:
        img_size = 224

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=img_size,
        min_size=197,
        data_format=K.image_data_format(),
        require_flatten=include_top,
    )

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # handle dimension ordering for different backends
    if K.common.image_dim_ordering() == "tf":
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3), name="conv1_zeropadding")(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name="conv1", use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name="bn_conv1")(x)
    x = Scale(axis=bn_axis, name="scale_conv1")(x)
    x = Activation("relu", name="conv1_relu")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name="pool1")(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block="a", strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="b")
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="c")

    x = conv_block(x, 3, [128, 128, 512], stage=3, block="a")
    for i in range(1, 8):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block="b" + str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block="a")
    for i in range(1, 36):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block="b" + str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block="a")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c")

    if large_input:
        x = AveragePooling2D((14, 14), name="avg_pool")(x)
    else:
        x = AveragePooling2D((7, 7), name="avg_pool")(x)

    # include classification layer by default, not included for feature extraction
    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation="softmax", name="fc1000")(x)
    else:
        if pooling == "avg":
            x = GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name="resnet152")

    # load weights
    if weights == "imagenet":
        if include_top:
            weights_path = get_file(
                "resnet152_weights_tf.h5",
                WEIGHTS_PATH,
                cache_subdir="models",
                md5_hash="cdb18a2158b88e392c0905d47dcef965",
            )
        else:
            weights_path = get_file(
                "resnet152_weights_tf_notop.h5",
                WEIGHTS_PATH_NO_TOP,
                cache_subdir="models",
                md5_hash="4a90dcdafacbd17d772af1fb44fc2660",
            )
        model.load_weights(weights_path, by_name=True)
        if K.backend() == "theano":
            layer_utils.convert_all_kernels_in_model(model)
            if include_top:
                maxpool = model.get_layer(name="avg_pool")
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name="fc1000")
                layer_utils.convert_dense_weights_data_format(
                    dense, shape, "channels_first"
                )

        if K.image_data_format() == "channels_first" and K.backend() == "tensorflow":
            warnings.warn(
                "You are using the TensorFlow backend, yet you "
                "are using the Theano "
                "image data format convention "
                '(`image_data_format="channels_first"`). '
                "For best performance, set "
                '`image_data_format="channels_last"` in '
                "your Keras config "
                "at ~/.keras/keras.json."
            )
    return model


if __name__ == "__main__":
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import tensorflow as tf

        tf.logging.set_verbosity(tf.logging.ERROR)
        import os
        os.makedirs("outputs", exist_ok=True)

        model = ResNet152(include_top=False, input_shape=(200, 200, 3), pooling="avg", weights="imagenet")
        model.save_weights("outputs/model.pkl")

"""
            with open(self.source_directory + "/" + self.train_py, "w") as file:
                file.write(create_model_py)

        assert os.path.isfile(self.source_directory + "/" + self.train_py), (
            f"The file {self.train_py} could not be found at "
            f"{self.source_directory}"
        )

    @classmethod
    def get_or_create_workspace(
        cls,
        configuration_file: str = project_configuration_file,
        train_py=train_py_default,
        score_py=score_py_default,
        **kwargs,
    ):
        """ Get or create a workspace if it doesn't exist.

        :param configuration_file: path to project configuration file. default: project.yml
        """
        project_configuration = ProjectConfiguration(configuration_file)
        assert project_configuration.has_value("subscription_id")
        assert project_configuration.has_value("resource_group")
        assert project_configuration.has_value("workspace_name")
        assert project_configuration.has_value("workspace_region")

        DeepRealtimeScore.create(
            subscription_id=project_configuration.get_value("subscription_id"),
            resource_group=project_configuration.get_value("resource_group"),
            name=project_configuration.get_value("workspace_name"),
            location=project_configuration.get_value("workspace_region"),
            exist_ok=True,
        )

        ws = DeepRealtimeScore(
            project_configuration.get_value("subscription_id"),
            project_configuration.get_value("resource_group"),
            project_configuration.get_value("workspace_name"),
            configuration_file,
            train_py=train_py,
            **kwargs,
        )
        return ws


class FPGARealtimeScore(RealtimeScoreAKSContext):
    @classmethod
    def create_aks(cls, ws: Workspace, aks_name, agent_count=1, location="eastus"):
        # Specify the Standard_PB6s Azure VM and location. Values for location may be "eastus", "southeastasia",
        # "westeurope", or "westus2". If no value is specified, the default is "eastus".
        prov_config = AksCompute.provisioning_configuration(
            vm_size="Standard_PB6s", agent_count=agent_count, location=location
        )
        if aks_name in ws.compute_targets:
            print("Existing AKS Found")
            return ComputeTarget(workspace=ws, name=aks_name)
        aks_target = ComputeTarget.create(
            workspace=ws, name=aks_name, provisioning_configuration=prov_config
        )
        aks_target.wait_for_completion(show_output=True)
        print(aks_target.provisioning_state)
        print(aks_target.provisioning_errors)

        return aks_target

    @classmethod
    def create_aks_service(
        cls,
        ws: Workspace,
        aks_target,
        image,
        aks_service_name="my-aks-service",
        autoscale_enabled=False,
        num_replicas=1,
        auth_enabled=False,
    ) -> AksWebservice:
        if aks_service_name in ws.webservices:
            return AksWebservice(ws, aks_service_name)

        # For this deployment, set the web service configuration without enabling auto-scaling
        # or authentication for testing
        aks_config = AksWebservice.deploy_configuration(
            autoscale_enabled=autoscale_enabled,
            num_replicas=num_replicas,
            auth_enabled=auth_enabled,
        )
        aks_service = AksWebservice.deploy_from_image(
            workspace=ws,
            name=aks_service_name,
            image=image,
            deployment_config=aks_config,
            deployment_target=aks_target,
        )
        aks_service.wait_for_deployment(show_output=True)
        return aks_service

    @classmethod
    def create_image(cls, converted_model, image_name, ws: Workspace):
        if image_name in ws.images:
            return Image(ws, image_name)
        image_config = AccelContainerImage.image_configuration()
        # Image name must be lowercase
        image = Image.create(
            name=image_name,
            models=[converted_model],
            image_config=image_config,
            workspace=ws,
        )
        image.wait_for_creation(show_output=False)
        return image

    @classmethod
    def register_resnet_50_model(
        cls, ws: Workspace, model_name: str, model_save_path: str, save_path: str
    ):
        # Input images as a two-dimensional tensor containing an arbitrary number of images represented a strings
        if model_name in ws.models:
            input_tensors, output_tensors = FPGARealtimeScore.get_resnet50_IO()
            return input_tensors, output_tensors, Model(ws, model_name)
        import tensorflow as tf

        tf.reset_default_graph()

        in_images = tf.placeholder(tf.string)
        image_tensors = utils.preprocess_array(in_images)
        print(image_tensors.shape)

        model_graph = QuantizedResnet50(save_path, is_frozen=True)
        feature_tensor = model_graph.import_graph_def(image_tensors)
        print(model_graph.version)
        print(feature_tensor.name)
        print(feature_tensor.shape)
        classifier_output = model_graph.get_default_classifier(feature_tensor)
        print(classifier_output)
        print("Saving model in {}".format(model_save_path))
        with tf.Session() as sess:
            model_graph.restore_weights(sess)
            tf.saved_model.simple_save(
                sess,
                model_save_path,
                inputs={"images": in_images},
                outputs={"output_alias": classifier_output},
            )
        input_tensors = in_images.name
        output_tensors = classifier_output.name
        print(input_tensors)
        print(output_tensors)
        registered_model = Model.register(
            workspace=ws, model_path=model_save_path, model_name=model_name
        )

        print(
            "Successfully registered: ",
            registered_model.name,
            registered_model.description,
            registered_model.version,
            sep="\t",
        )

        return input_tensors, output_tensors, registered_model

    @classmethod
    def convert_tf_model(cls, ws, input_tensors, output_tensors, registered_model):
        convert_request = AccelOnnxConverter.convert_tf_model(
            ws, registered_model, input_tensors, output_tensors
        )
        # If it fails, you can run wait_for_completion again with show_output=True.
        convert_request.wait_for_completion(show_output=False)
        # If the above call succeeded, get the converted model
        converted_model = convert_request.result
        print(
            "\nSuccessfully converted: ",
            converted_model.name,
            converted_model.url,
            converted_model.version,
            converted_model.id,
            converted_model.created_time,
            "\n",
        )
        return converted_model

    @classmethod
    def register_resnet_50(
        cls,
        ws: Workspace,
        model_name,
        image_name,
        save_path=os.path.expanduser("~/models"),
    ):
        if image_name in ws.images:
            return Image(ws, image_name)
        model_save_path = os.path.join(save_path, model_name)
        (
            input_tensors,
            output_tensors,
            registered_model,
        ) = FPGARealtimeScore.register_resnet_50_model(
            ws, model_name, model_save_path, save_path
        )
        converted_model = FPGARealtimeScore.convert_tf_model(
            ws, input_tensors, output_tensors, registered_model
        )
        image = FPGARealtimeScore.create_image(converted_model, image_name, ws)
        return image

    @classmethod
    def get_resnet50_IO(cls):
        import tensorflow as tf

        tf.reset_default_graph()
        in_images = tf.placeholder(tf.string)
        save_path = os.path.expanduser("~/models")
        model_graph = QuantizedResnet50(save_path, is_frozen=True)
        image_tensors = utils.preprocess_array(in_images)
        feature_tensor = model_graph.import_graph_def(image_tensors)
        classifier_output = model_graph.get_default_classifier(feature_tensor)
        input_tensors = in_images.name
        output_tensors = classifier_output.name
        return input_tensors, output_tensors

    @classmethod
    def get_prediction_client(cls, aks_service: AksWebservice):
        address = aks_service.scoring_uri
        ssl_enabled = address.startswith("https")
        address = address[address.find("/") + 2 :].strip("/")
        port = 443 if ssl_enabled else 80
        # Initialize Azure ML Accelerated Models client
        client = PredictionClient(
            address=address,
            port=port,
            use_ssl=ssl_enabled,
            service_name=aks_service.name,
        )
        return client


# def test_score_file(score_py):
#     exec(open(score_py).read())
#     exec("init()")
#     exec("response = run(MockRequest())")
#     exec("assert response")


class MockRequest:
    """Mock Request Class to create calls to test web service code"""

    method = "GET"

class MockImageRequest:
    """Mock Request Class to create calls to test web service code"""

    method = "POST"

    def __init__(self):
        self.files = {"image": open("snowleopardgaze.jpg", "rb")}

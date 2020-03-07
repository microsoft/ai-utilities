"""
AI-Utilities - realtime_score_context.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import json
import os
import time
from abc import ABC
from typing import Tuple, Any

from azure.mgmt.deploymentmanager.models import DeploymentMode
from azure.mgmt.resource import ResourceManagementClient
from azureml.contrib.functions import HTTP_TRIGGER, package
from azureml.core import ComputeTarget, Environment, Model, Webservice, Workspace, Image
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
from azure_utils.configuration.notebook_config import project_configuration_file, score_py_default, train_py_default
from azure_utils.configuration.project_configuration import ProjectConfiguration
from azure_utils.machine_learning.contexts.model_management_context import LocalTrainingContext, ModelManagementContext
from azure_utils.machine_learning.contexts.workspace_contexts import WorkspaceContext
from azure_utils.machine_learning.datasets.stack_overflow_data import clean_data, download_datasets, save_data, \
    split_duplicates
from azure_utils.machine_learning.deep.create_deep_model import get_or_create_resnet_image
from azure_utils.machine_learning.realtime.image import get_or_create_lightgbm_image, print_deployment_time, \
    print_image_deployment_info
from azure_utils.machine_learning.train_local import get_local_run_configuration
from azure_utils.notebook_widgets.workspace_widget import make_workspace_widget


class RealtimeScoreContext(WorkspaceContext):
    """ Real-time Score Context """

    def __init__(self, subscription_id, resource_group, workspace_name,
                 configuration_file: str = project_configuration_file, train_py="score.py", score_py="score.py",
                 settings_image_name="image_name", settings_aks_name="aks_name",
                 settings_aks_service_name="aks_service_name", **kwargs):
        super().__init__(subscription_id, resource_group, workspace_name, train_py=train_py, score_py=score_py,
                         **kwargs)
        project_configuration = ProjectConfiguration(configuration_file)
        self.project_configuration = project_configuration

        self.score_py = score_py

        self.settings_image_name = settings_image_name
        self.settings_aks_name = settings_aks_name
        self.settings_aks_service_name = settings_aks_service_name

        # Model Configuration
        self.model_tags = None
        self.model_description = None

        # Conda Configuration
        self.conda_file = None
        self.get_details()
        self.conda_pack = None
        self.requirements = None
        self.conda_env = CondaDependencies.create(conda_packages=self.conda_pack, pip_packages=self.requirements)

        # Env Configuration
        self.env_dependencies = None

        self.dockerfile = None
        self.image_dependencies = None
        self.image_description = None
        self.image_enable_gpu = False

        self.get_image = None
        self.aks_vm_size: str = "Standard_D4_v2"

        self.num_estimators = "1"

        self.node_count = 4
        self.num_replicas: int = 2
        self.cpu_cores: int = 1

        self.workspace_widget = None

    def test_service_local(self) -> None:
        """
        Test Scoring Service Locally by loading file
        """
        Model(self, self.model_name).download(exist_ok=True)
        exec(open(self.score_py).read())
        exec("init()")
        exec("response = run(MockRequest())")
        exec("assert response")

    def get_inference_config(self) -> InferenceConfig:
        """
        Get Infernce Configuration
        :return:
        """
        environment = Environment("conda-env")
        environment.python.conda_dependencies = self.conda_env

        inference_config = InferenceConfig(entry_script=self.score_py, environment=environment,
                                           source_directory="source")
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


class RealtimeScoreFunctionsContext(RealtimeScoreContext, ModelManagementContext, ABC):
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
        myenv.python.conda_dependencies = CondaDependencies.create(conda_packages=['scikit-learn'],
                                                                   pip_packages=['azureml-defaults'])
        return InferenceConfig(entry_script=self.score_py, environment=myenv)

    def get_or_create_function_image(self, config: InferenceConfig, models: list):
        """
        Create new configuration for deploying an scoring service to function image.

        :param config:
        :param models:
        """
        blob = package(self, models, config, functions_enabled=True, trigger=HTTP_TRIGGER)
        blob.wait_for_creation(show_output=True)
        # Display the package location/ACR path
        print(blob.location)


class RealtimeScoreAKSContext(RealtimeScoreContext):
    """
    Real-time Scoring with AKS Interface
    """

    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str, kwargs: dict):
        """
        Create new Real-time Scoring on AKS Context.

        :param subscription_id: Azure subscription id
        :param resource_group: Azure Resource Group name
        :param workspace_name: Azure Machine Learning Workspace
        :param kwargs: additional args
        """
        super().__init__(subscription_id, resource_group, workspace_name, **kwargs)
        self.aks_name = self.assert_and_get_value(self.settings_aks_name)
        self.aks_service_name = self.assert_and_get_value(self.settings_aks_service_name)
        self.aks_location = self.assert_and_get_value("workspace_region")

    @classmethod
    def get_or_or_create(cls, configuration_file: str = project_configuration_file, train_py=train_py_default,
                         score_py=score_py_default):
        """ Get or Create Real-time Endpoint on AKS

        :param configuration_file: path to project configuration file. default: project.yml
        :param train_py: python source file for training
        :param score_py: python source file for scoring
        """
        model, workspace = cls._get_workspace_and_model(configuration_file, score_py, train_py)
        inference_config = workspace.get_inference_config()
        aks_target = workspace.get_or_create_aks()
        web_service = workspace.get_or_create_aks_service(model, aks_target, inference_config)

        return workspace, web_service

    @classmethod
    def _get_workspace_and_model(cls, configuration_file: str, score_py: str, train_py: str) -> Tuple[Model, Any]:
        """
        Retrieve Workspace and Model

        :param configuration_file: path to project configuration file
        :param score_py: python source file for scoring
        :param train_py: python source file for training
        :return: Model and Workspace
        """
        workspace = cls.get_or_create_workspace(configuration_file, train_py=train_py, score_py=score_py)
        model = workspace.get_or_create_model()
        return model, workspace

    def get_or_create_aks(self) -> ComputeTarget:
        """
        Get or Create AKS Cluster

        :return: AKS Compute from Workspace
        """
        assert "_" not in self.project_configuration.get_value(
                self.settings_aks_service_name), self.settings_aks_service_name + " can not contain _"
        assert self.project_configuration.has_value("workspace_region")

        workspace_compute = self.compute_targets
        if self.aks_name in workspace_compute:
            return workspace_compute[self.aks_name]

        prov_config = AksCompute.provisioning_configuration(agent_count=self.node_count, vm_size=self.aks_vm_size,
                                                            location=self.aks_location)

        deploy_aks_start = time.time()
        aks_target = ComputeTarget.create(workspace=self, name=self.aks_name, provisioning_configuration=prov_config)

        aks_target.wait_for_completion(show_output=True)
        if self.show_output:
            service_name = "AKS"
            print_deployment_time(self.aks_service_name, deploy_aks_start, service_name)
            print(aks_target.provisioning_state)
            print(aks_target.provisioning_errors)
        aks_status = aks_target.get_status()
        assert aks_status == 'Succeeded', 'AKS failed to create'
        return aks_target

    def get_aks_deployment_config(self) -> AksServiceDeploymentConfiguration:
        """

        :return:
        """
        aks_deployment_configuration = {"num_replicas": self.num_replicas, "cpu_cores": self.cpu_cores,
                                        "enable_app_insights": True, "collect_model_data": True}
        return AksWebservice.deploy_configuration(**aks_deployment_configuration)

    def get_or_create_aks_service(self, model: Model, aks_target: AksCompute,
                                  inference_config: InferenceConfig) -> Webservice:
        """

        :param model:
        :param aks_target:
        :param inference_config:
        :return:
        """
        model_dict = model.serialize()

        if self._aks_exists():
            aks_service = self.get_web_service(self.aks_service_name)
            self._post_process_aks_deployment(aks_service, aks_target, model_dict)
            return aks_service

        aks_service = Model.deploy(self, self.aks_service_name, models=[model], inference_config=inference_config,
                                   deployment_target=aks_target, overwrite=True)
        self._post_process_aks_deployment(aks_service, aks_target, model_dict)

        try:
            if self.wait_for_completion:
                self.wait_then_configure_ping_test(aks_service, self.aks_service_name)
        finally:
            if self.show_output:
                print(aks_service.get_logs())
        return aks_service

    def _post_process_aks_deployment(self, aks_service: AksWebservice, aks_target: AksCompute, model_dict: dict):
        aks_dict = aks_service.serialize()
        self.workspace_widget = make_workspace_widget(model_dict, aks_dict)
        self.create_kube_config(aks_target)

    def wait_then_configure_ping_test(self, aks_service: AksWebservice, aks_service_name):
        """

        :param aks_service:
        :param aks_service_name:
        """
        aks_service.wait_for_deployment(show_output=self.show_output)
        self.configure_ping_test("ping-test-" + aks_service_name, self.get_details()['applicationInsights'],
                                 aks_service.scoring_uri, aks_service.get_keys()[0])

    def has_web_service(self, service_name: str) -> bool:
        """

        :param service_name:
        :return:
        """
        assert self.webservices
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
        return self.webservices[service_name]

    @staticmethod
    def create_kube_config(aks_target: AksCompute):
        """

        :param aks_target:
        """
        user_path = os.path.expanduser('~')
        kub_dir = os.path.join(user_path, '.kube')
        config_path = os.path.join(kub_dir, 'config')

        os.makedirs(kub_dir, exist_ok=True)
        with open(config_path, 'a') as f:
            f.write(aks_target.get_credentials()['userKubeConfig'])

    def _aks_exists(self) -> bool:
        """Check if Kubernetes Cluster Exists or has Failed"""
        aks_exists = self.has_web_service(self.aks_service_name) and self.get_web_service_state(
                self.aks_service_name) != "Failed"
        return aks_exists

    @staticmethod
    def configure_ping_test(ping_test_name: str, app_name: str, ping_url: str, ping_token: str):
        """

        :param ping_test_name:
        :param app_name:
        :param ping_url:
        :param ping_token:
        """
        project_configuration = ProjectConfiguration(project_configuration_file)
        assert project_configuration.has_value('subscription_id')
        credentials = AzureCliAuthentication()
        client = ResourceManagementClient(credentials, project_configuration.get_value('subscription_id'))
        template_path = os.path.join(os.path.dirname(__file__), 'templates', 'webtest.json')
        with open(template_path) as template_file_fd:
            template = json.load(template_file_fd)

        parameters = {'appName': app_name.split("components/")[1], 'pingURL': ping_url, 'pingToken': ping_token,
                      'location': project_configuration.get_value('workspace_region'),
                      'pingTestName': ping_test_name + "-" + project_configuration.get_value('workspace_region')}
        parameters = {k: {'value': v} for k, v in parameters.items()}

        deployment_properties = {'mode': DeploymentMode.incremental, 'template': template, 'parameters': parameters}

        deployment_async_operation = client.deployments.create_or_update(
                project_configuration.get_value('resource_group'), 'add-web-test', deployment_properties)
        deployment_async_operation.wait()


@deprecated(version='0.3.81', reason="Switch to using Env, this will be removed in 0.4.0")
class RealTimeScoreImageAndAKSContext(RealtimeScoreAKSContext):
    """ Old way to deploy AKS with Docker Image, use RealtimeScoreAKSContext instead"""

    @deprecated(version='0.3.81', reason="Switch to using Env, this will be removed in 0.4.0")
    def get_or_create_aks(self, aks_target) -> AksWebservice:
        """
        Get or Create AKS Service with new or existing Kubernetes Compute

        :param aks_target:
        :return: New or Existing Kubernetes Web Service
        """
        image_name = self.assert_and_get_value(self.settings_image_name)
        assert self.aks_vm_size

        if self._aks_exists():
            self.create_kube_config(aks_target)
            return self.get_web_service(self.aks_service_name)

        aks_config = self.get_aks_deployment_config()

        if image_name not in self.images:
            self.get_image()
        image = self.images[image_name]

        deploy_from_image_start = time.time()

        aks_service = Webservice.deploy_from_image(workspace=self, name=self.aks_service_name, image=image,
                                                   deployment_config=aks_config, deployment_target=aks_target,
                                                   overwrite=True)
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

    @deprecated(version='0.3.81', reason="Switch to using Env, this will be removed in 0.4.0")
    def get_inference_config(self, kwargs: dict) -> ContainerImageConfig:
        """ Get or Create new Docker Image Configuration for Machine Learning Workspace

        Image Configuration for running LightGBM in Azure Machine Learning Workspace

        :param kwargs: keyword args
        :return: new image configuration for Machine Learning Workspace
        """
        self.assert_image_params()

        self.write_conda_env()
        assert os.path.isfile(self.conda_file)

        self.image_tags['score_py_hash'] = self._get_file_md5(self.score_py)
        return ContainerImage.image_configuration(execution_script=self.score_py, runtime="python",
                                                  conda_file=self.conda_file, description=self.image_description,
                                                  dependencies=self.image_dependencies, docker_file=self.dockerfile,
                                                  tags=self.image_tags, enable_gpu=self.image_enable_gpu, **kwargs)

    @deprecated(version='0.3.81', reason="Switch to using Env, this will be removed in 0.4.0")
    def get_or_create_image(self, image_config: ContainerImageConfig, models: list = None) -> Image:
        """Get or Create new Docker Image from Machine Learning Workspace

        :param image_config:
        :param models:
        """
        if not models:
            models = []

        image_name = self.assert_and_get_value(self.settings_image_name)

        if image_name in self.images and self.images[image_name].creation_state != "Failed":
            # hasher = hashlib.md5()
            # with open(self.score_py, 'rb') as afile:
            #     buf = afile.read()
            #     hasher.update(buf)
            # if "hash" in Image(workspace, image_name).tags \
            #         and hasher.hexdigest() == Image(workspace, image_name).tags['hash']:
            return self.images[image_name]

        image_create_start = time.time()
        image = ContainerImage.create(name=image_name, models=models, image_config=image_config, workspace=self)
        image.wait_for_creation(show_output=self.show_output)
        assert image.creation_state != "Failed"
        if self.show_output:
            print_image_deployment_info(image, image_name, image_create_start)
        return image


class MLRealtimeScore(RealtimeScoreAKSContext, RealtimeScoreFunctionsContext, LocalTrainingContext):
    """ Light GBM Real Time Scoring"""

    def __init__(self, subscription_id, resource_group, workspace_name, run_configuration,
                 configuration_file: str = project_configuration_file, train_py=train_py_default,
                 score_py=score_py_default, conda_file="my_env.yml"):
        super().__init__(subscription_id, resource_group, workspace_name, get_local_run_configuration(),
                         configuration_file, train_py=train_py, score_py=score_py)
        self.get_docker_file()
        self.setting_image_name = "image_name"
        self.settings_aks_name = "aks_name"
        self.settings_aks_service_name = "aks_service_name"

        self.execution_script = "score.py"
        with open(self.execution_script, 'w') as file:
            file.write("""        
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


def run(body):
    logger = logging.getLogger("scoring_script")
    logger.info("run")
    return json.dumps({'call': True})

""")

        self.model_name = "question_match_model"
        self.experiment_name = "mlaks-train-on-local"
        self.model_path = "./outputs/model.pkl"

        self.source_directory = "./script"
        self.script = "create_model.py"
        if not os.path.isfile(self.source_directory + "/" + self.script):
            os.makedirs("script", exist_ok=True)

            create_model_py = "from azure_utils.machine_learning import create_model\n\nif __name__ == '__main__':\n" \
                              "    create_model.main()"
            with open(self.source_directory + "/" + self.script, "w") as file:
                file.write(create_model_py)

        self.show_output = True
        self.args = ["--inputs", os.path.abspath(directory + "/data_folder"), "--outputs", "outputs", "--estimators",
                     self.num_estimators, "--match", "2", ]
        self.run_configuration = run_configuration

        self.test_size = 0.21
        self.min_text = 150
        self.min_dupes = 12
        self.match = 20

        self.prepare_data()
        # Image Configuration
        self.get_image = get_or_create_lightgbm_image
        self.image_tags = {"area": "text", "type": "lightgbm", "name": "AKS", "project": "AML"}
        self.image_description = "Image for lightgbm model"
        self.image_dependencies = None
        self.image_enable_gpu = False

        self.conda_file = conda_file
        self.conda_pack = ["scikit-learn==0.19.1", "pandas==0.23.3"]
        self.requirements = ["lightgbm==2.1.2", "azureml-defaults==1.0.57", "azureml-contrib-services",
                             "opencensus-ext-flask", "Microsoft-AI-Azure-Utility-Samples"]
        self.conda_env = CondaDependencies.create(conda_packages=self.conda_pack, pip_packages=self.requirements)

    def get_docker_file(self) -> None:
        """
        Create Docker File with GCC
        """
        self.dockerfile = "dockerfile"
        with open(self.dockerfile, "w") as file:
            file.write("RUN apt update -y && apt upgrade -y && apt install -y build-essential")

    def prepare_data(self) -> None:
        """
        Prepare the training data
        """
        outputs_path = directory + "/data_folder"
        dupes_test_path = os.path.join(outputs_path, "dupes_test.tsv")
        questions_path = os.path.join(outputs_path, "questions.tsv")

        if not (os.path.isfile(dupes_test_path) and os.path.isfile(questions_path)):
            answers, dupes, questions = download_datasets(show_output=self.show_output)

            # Clean up all text, and keep only data with some clean text.
            dupes, label_column, questions = clean_data(answers, dupes, self.min_dupes, self.min_text, questions,
                                                        self.show_output)

            # Split dupes into train and test ensuring at least one of each label class is in test.
            balanced_pairs_test, balanced_pairs_train, dupes_test = split_duplicates(dupes, label_column, self.match,
                                                                                     questions, self.show_output,
                                                                                     self.test_size)

            save_data(balanced_pairs_test, balanced_pairs_train, dupes_test, dupes_test_path, outputs_path, questions,
                      questions_path, self.show_output)


class DeepRealtimeScore(RealtimeScoreAKSContext, RealtimeScoreFunctionsContext, LocalTrainingContext):
    """ Resnet Real-time Scoring"""

    def __init__(self, subscription_id, resource_group, workspace_name,
                 configuration_file: str = project_configuration_file, train_py=train_py_default,
                 score_py=score_py_default, conda_file="my_env.yml", setting_image_name="deep_image_name",
                 settings_aks_name="deep_aks_name", settings_aks_service_name="deep_aks_service_name"):
        super().__init__(subscription_id, resource_group, workspace_name, get_local_run_configuration(),
                         configuration_file, train_py=train_py, score_py=score_py,
                         setting_image_name=setting_image_name, settings_aks_name=settings_aks_name,
                         settings_aks_service_name=settings_aks_service_name)
        self.setting_image_name = "deep_image_name"
        self.settings_aks_name = "deep_aks_name"
        self.settings_aks_service_name = "deep_aks_service_name"

        # Experiment Configuration
        self.experiment_name = "dlrts-train-on-local"

        # Model Configuration
        self.model_name = "resnet_model_2"
        self.model_tags = {"model": "dl", "framework": "resnet"}
        self.model_description = "resnet 152 model"
        self.model_path = "./outputs/model.pkl"

        # Conda Configuration
        self.conda_file = conda_file
        self.write_conda_env()
        self.conda_pack = ["tensorflow-gpu==1.14.0"]
        self.requirements = ["keras==2.2.0", "Pillow==5.2.0", "azureml-defaults", "azureml-contrib-services",
                             "toolz==0.9.0", "git+https://github.com/microsoft/AI-Utilities.git@deep_learning_2"]
        self.conda_env = CondaDependencies.create(conda_packages=self.conda_pack, pip_packages=self.requirements)

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


class MockRequest:
    """Mock Request Class to create calls to test web service code"""
    method = 'GET'

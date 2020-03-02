"""
AI-Utilities - ai_workspace.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import time

from azureml.core import Workspace, Model, Webservice, ComputeTarget
from azureml.core.compute import AksCompute
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.image import ContainerImage
from azureml.core.webservice import AksWebservice

from azure_utils.configuration.notebook_config import project_configuration_file
from azure_utils.configuration.project_configuration import ProjectConfiguration
from azure_utils.machine_learning.datasets.stack_overflow_data import create_stack_overflow_data
from azure_utils.machine_learning.deep.create_deep_model import create_deep_model, develop_model_driver, \
    get_or_create_resnet_image
from azure_utils.machine_learning.realtime.image import get_or_create_lightgbm_image
from azure_utils.machine_learning.train_local import train_local, create_stack_overflow_model_script
from azure_utils.machine_learning.utils import get_or_create_workspace_from_project


class AILabWorkspace(Workspace):
    """ AI Workspace """
    image_settings_name = "image_name"
    settings_aks_name = "aks_name"
    settings_aks_service_name = "aks_service_name"

    def __init__(self, subscription_id, resource_group, workspace_name):
        super().__init__(subscription_id, resource_group, workspace_name)
        self.tags = None
        self.dockerfile = None
        self.dependencies = None
        self.description = None
        self.conda_file = None
        self.execution_script = None
        self.enable_gpu = False

        self.get_image = None
        self.setting_service_name = "aks_service_name"
        self.setting_image_name = "image_name"
        self.vm_size: str = "Standard_D4_v2"

    @classmethod
    def get_or_or_create_realtime_endpoint(cls, **kwargs):
        """ Get or Create Real-time Endpoint

        :param kwargs: keyword args
        """
        workspace = cls.get_or_create_workspace()
        models = [cls.get_or_create_model()]
        config = workspace.get_or_create_image_configuration()
        cls.get_or_create_image(config, cls.image_settings_name, models=models)
        return workspace.get_or_create_service()

    @classmethod
    def get_or_create_workspace(cls, configuration_file: str = project_configuration_file):
        """ Get or create a workspace if it doesn't exist.

        :param configuration_file:
        """
        project_configuration = ProjectConfiguration(configuration_file)

        return cls(project_configuration.get_value('subscription_id'),
                   project_configuration.get_value('resource_group'),
                   project_configuration.get_value('workspace_name'))

    def get_or_create_image_configuration(self, **kwargs):
        """ Get or Create new Docker Image Configuration for Machine Learning Workspace

        :param kwargs: keyword args
        """
        """
        Image Configuration for running LightGBM in Azure Machine Learning Workspace

        :return: new image configuration for Machine Learning Workspace
        """
        assert self.execution_script
        assert self.conda_file
        assert self.description
        assert self.tags

        return ContainerImage.image_configuration(execution_script=self.execution_script, runtime="python",
                                                  conda_file=self.conda_file, description=self.description,
                                                  dependencies=self.dependencies, docker_file=self.dockerfile,
                                                  tags=self.tags, enable_gpu=self.enable_gpu)

    def get_or_create_aks(self, configuration_file: str = project_configuration_file, vm_size: str = "Standard_D4_v2",
                          node_count: int = 4, show_output: bool = True, **kwargs):
        """
        Get or Create AKS Cluster

        :param configuration_file:
        :param kwargs: keyword args
        :return: AKS Compute from Workspace
        """
        project_configuration = ProjectConfiguration(configuration_file)
        assert project_configuration.has_settings(self.settings_aks_name)
        assert project_configuration.has_settings(self.settings_aks_service_name)
        assert "_" not in project_configuration.get_value(
            self.settings_aks_service_name), self.settings_aks_service_name + " can not contain _"
        assert project_configuration.has_settings("workspace_region")

        aks_name = project_configuration.get_value(self.settings_aks_name)
        aks_service_name = project_configuration.get_value(self.settings_aks_service_name)
        aks_location = project_configuration.get_value("workspace_region")

        workspace = get_or_create_workspace_from_project(project_configuration)
        workspace_compute = workspace.compute_targets
        if aks_name in workspace_compute:
            return workspace_compute[aks_name]

        prov_config = AksCompute.provisioning_configuration(agent_count=node_count, vm_size=vm_size,
                                                            location=aks_location)

        deploy_aks_start = time.time()
        aks_target = ComputeTarget.create(workspace=workspace, name=aks_name, provisioning_configuration=prov_config)

        aks_target.wait_for_completion(show_output=True)
        if show_output:
            deployment_time_secs = str(time.time() - deploy_aks_start)
            print("Deployed AKS with name "
                  + aks_service_name + ". Took " + deployment_time_secs + " seconds.")
            print(aks_target.provisioning_state)
            print(aks_target.provisioning_errors)
        aks_status = aks_target.get_status()
        assert aks_status == 'Succeeded', 'AKS failed to create'
        return aks_target

    def get_or_create_service(self, configuration_file: str = project_configuration_file,
                              node_count: int = 4, num_replicas: int = 2,
                              cpu_cores: int = 1, show_output: bool = True, **kwargs) -> AksWebservice:
        """
        Get or Create AKS Service with new or existing Kubernetes Compute

        :param configuration_file: path to project configuration file. default: project.yml
        :param node_count: number of nodes in Kubernetes cluster. default: 4
        :param num_replicas: number of replicas in Kubernetes cluster. default: 2
        :param cpu_cores: cpu cores for web service. default: 1
        :param show_output: toggle on/off standard output. default: `True`
        :return: New or Existing Kubernetes Web Service
        """
        aks_target = self.get_or_create_aks(configuration_file=configuration_file, vm_size=self.vm_size,
                                            node_count=node_count, show_output=show_output)

        project_configuration = ProjectConfiguration(configuration_file)

        assert project_configuration.has_settings(self.setting_service_name)
        assert project_configuration.has_settings(self.setting_image_name)

        aks_service_name = project_configuration.get_value(self.setting_service_name)
        image_name = project_configuration.get_value(self.setting_image_name)

        workspace = get_or_create_workspace_from_project(project_configuration, show_output=show_output)

        if aks_service_name in workspace.webservices:
            return workspace.webservices[aks_service_name]

        aks_config = AksWebservice.deploy_configuration(num_replicas=num_replicas, cpu_cores=cpu_cores)

        if image_name not in workspace.images:
            self.get_image()
        image = workspace.images[image_name]

        deploy_from_image_start = time.time()
        aks_service = Webservice.deploy_from_image(workspace=workspace, name=aks_service_name, image=image,
                                                   deployment_config=aks_config, deployment_target=aks_target)
        aks_service.wait_for_deployment(show_output=show_output)
        if show_output:
            deployment_time_secs = str(time.time() - deploy_from_image_start)
            print("Deployed Image with name "
                  + aks_service_name + ". Took " + deployment_time_secs + " seconds.")
            print(aks_service.state)
            print(aks_service.get_logs())
        return aks_service

    @staticmethod
    def get_or_create_model(**kwargs):
        """Get Or Create Model

        :param kwargs: keyword args
        """
        raise NotImplementedError

    @staticmethod
    def get_or_create_image(image_config, image_settings_name, models=None, show_output=True,
                            configuration_file: str = project_configuration_file, **kwargs):
        """Get or Create new Docker Image from Machine Learning Workspace

        :param image_config:
        :param image_settings_name:
        :param models:
        :param show_output:
        :param configuration_file:
        :param kwargs: keyword args
        """
        project_configuration = ProjectConfiguration(configuration_file)

        if not models:
            models = []

        assert project_configuration.has_settings(image_settings_name)
        image_name = project_configuration.get_value(image_settings_name)

        workspace = get_or_create_workspace_from_project(project_configuration, show_output=show_output)

        workspace_images = workspace.images
        if image_name in workspace_images and workspace_images[image_name].creation_state != "Failed":
            return workspace_images[image_name]

        image_create_start = time.time()
        image = ContainerImage.create(name=image_name, models=models, image_config=image_config,
                                      workspace=workspace)
        image.wait_for_creation(show_output=show_output)
        assert image.creation_state != "Failed"
        if show_output:
            deployment_time_secs = str(time.time() - image_create_start)
            print("Deployed Image with name " + image_name + ". Took " + deployment_time_secs + " seconds.")
            print(image.name)
            print(image.version)
            print(image.image_build_log_uri)
        return image


class RTSWorkspace(AILabWorkspace):
    """ Light GBM Real Time Scoring"""
    image_settings_name = "image_name"

    def __init__(self, subscription_id, resource_group, workspace_name, conda_file="lgbmenv.yml",
                 execution_script="score.py"):
        super().__init__(subscription_id, resource_group, workspace_name)
        conda_pack = [
            "scikit-learn==0.19.1",
            "pandas==0.23.3"
        ]
        requirements = [
            "lightgbm==2.1.2",
            "azureml-defaults==1.0.57",
            "azureml-contrib-services",
            "Microsoft-AI-Azure-Utility-Samples"
        ]
        lgbmenv = CondaDependencies.create(conda_packages=conda_pack, pip_packages=requirements)
        self.conda_file = conda_file
        self.execution_script = execution_script
        self.dependencies = None
        with open(conda_file, "w") as file:
            file.write(lgbmenv.serialize_to_string())

        self.dockerfile = "dockerfile"
        with open(self.dockerfile, "w") as file:
            file.write("RUN apt update -y && apt upgrade -y && apt install -y build-essential")

        with open(execution_script, 'w') as file:
            file.write("""        
        import json
        import logging


        def init():
            logger = logging.getLogger("scoring_script")
            logger.info("init")


        def run(body):
            logger = logging.getLogger("scoring_script")
            logger.info("run")
            return json.dumps({'call': True})
        """)
        self.description = "Image with lightgbm model"
        self.tags = {"area": "text", "type": "lightgbm"}
        self.get_image = get_or_create_lightgbm_image

    @staticmethod
    def get_or_create_model(**kwargs) -> Model:
        """
        Get or Create Model

        :param kwargs: keyword args
        :return: Model from Workspace
        """

        create_stack_overflow_data()
        create_stack_overflow_model_script()
        return train_local(**kwargs)

    @staticmethod
    def create_stack_overflow_data():
        """Download and crate Stack Overflow QA Dataset"""
        return create_stack_overflow_data()

    @staticmethod
    def create_stack_overflow_model_script():
        """Create Script to Train LightGBM on Stack Overflow Data"""
        return create_stack_overflow_model_script()

    @staticmethod
    def train_local(**kwargs) -> Model:
        """
        Train Model Locally

        :param kwargs: keyword args
        :return: Model trained Locally
        """
        return train_local(**kwargs)


class DeepRTSWorkspace(AILabWorkspace):
    """ Resnet Real-time Scoring"""
    image_settings_name = "mydeepimage"
    settings_aks_name = "deep_aks_name"

    def __init__(self, subscription_id, resource_group, workspace_name, execution_script="driver.py",
                 conda_file="img_env.yml"):
        super().__init__(subscription_id, resource_group, workspace_name)
        conda_pack = [
            "tensorflow-gpu==1.14.0"
        ]
        requirements = [
            "keras==2.2.0",
            "Pillow==5.2.0",
            "azureml-defaults",
            "azureml-contrib-services",
            "toolz==0.9.0"
        ]
        imgenv = CondaDependencies.create(conda_packages=conda_pack, pip_packages=requirements)
        self.conda_file = conda_file
        with open(conda_file, "w") as file:
            file.write(imgenv.serialize_to_string())

        self.description = "Image for AKS Deployment Tutorial"
        self.execution_script = execution_script
        self.dependencies = ["resnet152.py"]
        self.tags = {"name": "AKS", "project": "AML"}
        self.enable_gpu = True

        self.get_image = get_or_create_resnet_image
        self.setting_service_name = "deep_aks_service_name"
        self.setting_image_name = "deep_image_name"
        self.vm_size = "Standard_NC6"

    @staticmethod
    def get_or_create_model(**kwargs) -> Model:
        """
        Get or Create Model

        :param kwargs: keyword args
        :return: Model from Workspace
        """
        return create_deep_model()

    @staticmethod
    def get_or_create_model_driver():
        """ Get or Create Model Driver """
        develop_model_driver()

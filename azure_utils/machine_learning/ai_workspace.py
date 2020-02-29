"""
AI-Utilities - ai_workspace.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from azureml.core import Workspace, Image, Model
from azureml.core.compute import AksCompute
from azureml.core.webservice import AksWebservice

from azure_utils.machine_learning.datasets.stack_overflow_data import create_stack_overflow_data
from azure_utils.machine_learning.deep.create_deep_model import create_deep_model, develop_model_driver, build_image, \
    deploy_on_aks, get_or_create_deep_aks
from azure_utils.machine_learning.realtime.image import get_or_create_image
from azure_utils.machine_learning.realtime.kubernetes import get_or_create_aks_service, get_or_create_aks
from azure_utils.machine_learning.train_local import train_local, create_stack_overflow_model_script
from azure_utils.machine_learning.utils import get_or_create_workspace_from_file


class AILabWorkspace(Workspace):
    """ AI Workspace """

    def __init__(self, subscription_id, resource_group, workspace_name):
        super().__init__(subscription_id, resource_group, workspace_name)

    @staticmethod
    def get_or_or_create_realtime_endpoint(**kwargs):
        """ Get or Create Real-time Endpoint

        :param kwargs: keyword args
        """
        raise NotImplementedError

    @staticmethod
    def get_or_create_workspace(**kwargs):
        """ Get or create a workspace if it doesn't exist.

        :param kwargs: keyword args
        """
        return get_or_create_workspace_from_file(**kwargs)

    @staticmethod
    def get_or_create_image(**kwargs):
        """ Get or Create new Docker Image from Machine Learning Workspace

        :param kwargs: keyword args
        """
        raise NotImplementedError

    @staticmethod
    def get_or_create_aks(**kwargs):
        """ Get or Create Azure Machine Learning Kubernetes Compute

        :param kwargs: keyword args
        """
        raise NotImplementedError

    @staticmethod
    def get_or_create_service(**kwargs):
        """Get Or Create Kubernetes Compute and Web Service

        :param kwargs: keyword args
        """
        raise NotImplementedError

    @staticmethod
    def get_or_create_model(**kwargs):
        """Get Or Create Model

        :param kwargs: keyword args
        """
        raise NotImplementedError


class RTSWorkspace(AILabWorkspace):
    """ Light GBM Real Time Scoring"""

    @staticmethod
    def get_or_create_model(**kwargs) -> Model:
        """
        Get or Create Model

        :param kwargs: keyword args
        :return: Model from Workspace
        """
        RTSWorkspace.create_stack_overflow_data()
        RTSWorkspace.create_stack_overflow_model_script()
        return RTSWorkspace.train_local()

    @staticmethod
    def get_or_create_image(**kwargs) -> Image:
        """ Get or Create new Docker Image from Machine Learning Workspace

        :param kwargs: keyword args
        :return: Image from Workspace
        """
        return get_or_create_image(**kwargs)

    @staticmethod
    def get_or_create_service(**kwargs) -> AksWebservice:
        """Get Or Create Kubernetes Compute and Web Service

        :param kwargs: keyword args
        :return: AKS Web Service from Workspace
        """
        return get_or_create_aks_service(**kwargs)

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

    @staticmethod
    def get_or_create_aks(**kwargs) -> AksCompute:
        """
        Get or Create AKS Cluster

        :param kwargs: keyword args
        :return: AKS Compute from Workspace
        """
        return get_or_create_aks()

    @staticmethod
    def get_or_or_create_realtime_endpoint(**kwargs):
        """ Get or Create Real-time Endpoint

        :param kwargs: keyword args
        """
        RTSWorkspace.get_or_create_workspace()
        RTSWorkspace.get_or_create_model()
        RTSWorkspace.get_or_create_image()
        RTSWorkspace.get_or_create_service()


class DeepRTSWorkspace(AILabWorkspace):
    """ Resnet Real-time Scoring"""

    @staticmethod
    def get_or_create_model(**kwargs):
        """
        Get or Create Model

        :param kwargs: keyword args
        :return: Model from Workspace
        """
        create_deep_model()

    @staticmethod
    def get_or_create_image(**kwargs):
        """ Get or Create new Docker Image from Machine Learning Workspace

        :param kwargs: keyword args
        :return: Image from Workspace
        """
        build_image(**kwargs)

    @staticmethod
    def get_or_create_aks(**kwargs):
        """
        Get or Create AKS Cluster

        :param kwargs: keyword args
        :return: AKS Compute from Workspace
        """
        get_or_create_deep_aks(**kwargs)

    @staticmethod
    def get_or_create_service(**kwargs):
        """Get Or Create Kubernetes Compute and Web Service

        :param kwargs: keyword args
        :return: AKS Web Service from Workspace
        """
        deploy_on_aks(**kwargs)

    @staticmethod
    def get_or_create_model_driver():
        """ Get or Create Model Driver """
        develop_model_driver()

    @staticmethod
    def get_or_or_create_realtime_endpoint(**kwargs):
        """ Get or Create Real-time Endpoint

        :param kwargs: keyword args
        """
        DeepRTSWorkspace.get_or_create_workspace()
        DeepRTSWorkspace.get_or_create_model()
        DeepRTSWorkspace.get_or_create_image()
        DeepRTSWorkspace.get_or_create_service()

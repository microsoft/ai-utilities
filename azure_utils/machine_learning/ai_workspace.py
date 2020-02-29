"""
AI-Utilities - ai_workspace.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from azureml.core import Workspace

from azure_utils.machine_learning.datasets.stack_overflow_data import create_stack_overflow_data
from azure_utils.machine_learning.deep.create_deep_model import create_deep_model, develop_model_driver, build_image, \
    deploy_on_aks, get_or_create_deep_aks
from azure_utils.machine_learning.realtime.image import get_or_create_image
from azure_utils.machine_learning.realtime.kubernetes import get_or_create_aks_service, get_or_create_aks
from azure_utils.machine_learning.train_local import train_local, create_stack_overflow_model_script
from azure_utils.machine_learning.utils import get_or_create_workspace_from_file


class AILabWorkspace(Workspace):
    """ AI Workspace """

    def __init__(self, ws):
        super().__init__(ws._subscription_id, ws._resource_group, ws._workspace_name)
        print("Created Workspace")

    @staticmethod
    def get_or_or_create_realtime_endpoint():
        raise NotImplementedError

    @staticmethod
    def get_or_create_workspace(**kwargs):
        """Get or create a workspace if it doesn't exist."""
        return AILabWorkspace(get_or_create_workspace_from_file(**kwargs))

    @staticmethod
    def get_or_create_image(**kwargs):
        """ Get or Create new Docker Image from Machine Learning Workspace """
        raise NotImplementedError

    @staticmethod
    def get_or_create_aks(**kwargs):
        """ Get or Create Azure Machine Learning Kubernetes Compute"""
        raise NotImplementedError

    @staticmethod
    def get_or_create_service(**kwargs):
        """Get Or Create Kubernetes Compute and Web Service"""
        raise NotImplementedError

    @staticmethod
    def get_or_create_model():
        raise NotImplementedError


class RTSWorkspace(AILabWorkspace):
    def __init__(self, ws):
        super().__init__(ws)

    @staticmethod
    def get_or_create_model():
        RTSWorkspace.create_stack_overflow_data()
        RTSWorkspace.create_stack_overflow_model_script()
        RTSWorkspace.train_local()

    @staticmethod
    def get_or_create_image(**kwargs):
        """ Get or Create new Docker Image from Machine Learning Workspace """
        return get_or_create_image(**kwargs)

    @staticmethod
    def get_or_create_service(**kwargs):
        """Get Or Create Kubernetes Compute and Web Service"""
        return get_or_create_aks_service(**kwargs)

    @staticmethod
    def create_stack_overflow_data():
        return create_stack_overflow_data()

    @staticmethod
    def create_stack_overflow_model_script():
        return create_stack_overflow_model_script()

    @staticmethod
    def train_local():
        return train_local()

    @staticmethod
    def get_or_create_aks(**kwargs):
        return get_or_create_aks()

    @staticmethod
    def get_or_or_create_realtime_endpoint():
        RTSWorkspace.get_or_create_workspace()
        RTSWorkspace.get_or_create_model()
        RTSWorkspace.get_or_create_image()
        RTSWorkspace.get_or_create_service()


class DeepRTSWorkspace(AILabWorkspace):

    @staticmethod
    def get_or_create_model():
        create_deep_model()

    @staticmethod
    def get_or_create_image(**kwargs):
        build_image(**kwargs)

    @staticmethod
    def get_or_create_aks():
        get_or_create_deep_aks()

    @staticmethod
    def get_or_create_service():
        deploy_on_aks()

    @staticmethod
    def get_or_create_model_driver():
        develop_model_driver()

    @staticmethod
    def get_or_or_create_realtime_endpoint():
        DeepRTSWorkspace.get_or_create_workspace()
        DeepRTSWorkspace.get_or_create_model()
        DeepRTSWorkspace.get_or_create_image()
        DeepRTSWorkspace.get_or_create_service()

"""
AI-Utilities - ai_workspace.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import hashlib
import warnings

from azureml._base_sdk_common.common import check_valid_resource_name
from azureml.core import Workspace
from azureml.exceptions import UserErrorException

from azure_utils.configuration.notebook_config import (
    project_configuration_file,
    score_py_default,
    train_py_default,
)
from azure_utils.configuration.project_configuration import ProjectConfiguration


class WorkspaceContext(Workspace):
    """
    AzureML Workspace Context - Base Framework Interface
    """

    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        workspace_name: str, configuration_file: str = project_configuration_file, project_configuration=None,
        train_py: str = train_py_default,
        score_py: str = score_py_default,
        **kwargs
    ):
        """
        Interface Constructor for Workspace Context

        :param subscription_id: Azure subscription id
        :param resource_group: Azure Resource Group name
        :param workspace_name: Azure Machine Learning Workspace
        :param configuration_file: path to project configuration file. default: project.yml
        :param train_py: python source file for training
        :param score_py: python source file for scoring
        """
        super().__init__(subscription_id, resource_group, workspace_name, **kwargs)
        if not project_configuration:
            self.project_configuration = ProjectConfiguration(configuration_file)

        self.image_tags = None
        self.args = None
        self.train_py = train_py
        self.score_py = score_py
        self.show_output = True
        self.source_directory = "./script"
        self.experiment_name = None
        self.model_name = None
        self.wait_for_completion = True
        self.model_path = None

    @classmethod
    def get_or_create_workspace(
        cls,
        configuration_file: str = project_configuration_file,
        project_configuration: ProjectConfiguration = None,
        **kwargs
    ):
        """ Get or create a workspace if it doesn't exist.

        :param configuration_file:
        :param project_configuration: ProjectConfiguration
        """
        if not project_configuration:
            project_configuration = ProjectConfiguration(configuration_file)
        assert project_configuration.has_value("subscription_id")
        assert project_configuration.has_value("resource_group")
        assert project_configuration.has_value("workspace_name")
        assert project_configuration.has_value("workspace_region")

        try:
            check_valid_resource_name(
                project_configuration.get_value("workspace_name"), "Workspace"
            )
        except UserErrorException:
            print(project_configuration.get_value("workspace_name"))
            raise

        cls.create(
            subscription_id=project_configuration.get_value("subscription_id"),
            resource_group=project_configuration.get_value("resource_group"),
            name=project_configuration.get_value("workspace_name"),
            location=project_configuration.get_value("workspace_region"),
            exist_ok=True,
        )

        ws = cls(
            subscription_id=project_configuration.get_value("subscription_id"),
            resource_group=project_configuration.get_value("resource_group"),
            workspace_name=project_configuration.get_value("workspace_name"),
            project_configuration=project_configuration,
            **kwargs
        )
        return ws

    @staticmethod
    def _get_file_md5(file_name: str) -> str:
        hasher = hashlib.md5()
        with open(file_name, "rb") as afile:
            buf = afile.read()
            hasher.update(buf)
        file_hash = hasher.hexdigest()
        return file_hash

    def assert_and_get_value(self, setting_name: str) -> str:
        """

        :param setting_name:
        :return:
        """
        assert self.project_configuration.has_value(setting_name)
        return self.project_configuration.get_value(setting_name)

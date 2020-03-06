"""
AI-Utilities - ai_workspace.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from azureml.core import Workspace

from azure_utils.configuration.notebook_config import project_configuration_file, train_py_default, score_py_default
from azure_utils.configuration.project_configuration import ProjectConfiguration


class WorkspaceContext(Workspace):
    def __init__(self, subscription_id, resource_group, workspace_name,
                 configuration_file: str = project_configuration_file,
                 train_py=train_py_default, score_py=score_py_default):
        super().__init__(subscription_id, resource_group, workspace_name)
        self.configuration_file = configuration_file
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
    def get_or_create_workspace(cls, configuration_file: str = project_configuration_file,
                                train_py=train_py_default, score_py=score_py_default):
        """ Get or create a workspace if it doesn't exist.

        :param score_py:
        :param train_py:
        :param configuration_file:
        """
        project_configuration = ProjectConfiguration(configuration_file)
        assert project_configuration.has_value('subscription_id')
        assert project_configuration.has_value('resource_group')
        assert project_configuration.has_value('workspace_name')
        assert project_configuration.has_value('workspace_region')

        cls.create(subscription_id=project_configuration.get_value('subscription_id'),
                   resource_group=project_configuration.get_value('resource_group'),
                   name=project_configuration.get_value('workspace_name'),
                   location=project_configuration.get_value('workspace_region'),
                   create_resource_group=True, exist_ok=True)

        ws = cls(project_configuration.get_value('subscription_id'),
                 project_configuration.get_value('resource_group'),
                 project_configuration.get_value('workspace_name'),
                 configuration_file, train_py=train_py, score_py=score_py)
        return ws

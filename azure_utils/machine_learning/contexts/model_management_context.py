"""
AI-Utilities - model_management_context.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import os
from abc import ABC

from azureml.core import Experiment, Model, ScriptRunConfig, Run
from azureml.exceptions import ActivityFailedException

from azure_utils.configuration.notebook_config import (
    project_configuration_file,
    train_py_default,
)
from azure_utils.machine_learning.contexts.workspace_contexts import WorkspaceContext
from azure_utils.machine_learning.train_local import get_local_run_configuration


class ModelManagementContext(WorkspaceContext):
    """
    Interface for Contexts that require Model Management
    """

    def __init__(
        self,
        subscription_id,
        resource_group,
        workspace_name,
        run_configuration,
        configuration_file: str = project_configuration_file,
        train_py=train_py_default,
    ):
        super().__init__(
            subscription_id,
            resource_group,
            workspace_name,
            configuration_file=configuration_file,
            train_py=train_py,
        )
        self.configuration_file = configuration_file
        self.run_configuration = run_configuration
        self.model_name = None
        self.wait_for_completion = True
        self.model_path = None

    def get_or_create_model(self) -> Model:
        """
        Get or Create Model

        :return: Model from Workspace
        """
        assert self.model_name

        print("Check if Model exists.")
        if self.model_name in self.models:
            print("Model does exists.")
            # if get_model(self.model_name).tags['train_py_hash'] == self.get_file_md5(
            #         self.source_directory + "/" + self.script):
            model = Model(self, name=self.model_name)
            if not os.path.isdir("outputs"):
                model.download("outputs", exist_ok=True)
            return model
        print("Model does not exists.")
        model = self.train_model()

        assert model
        if self.show_output:
            print(model.name, model.version, model.url, sep="\n")
        return model

    def train_model(self) -> Model:
        """
        Train Model with Experiment Run

        :return: registered model from Experiment run.
        """
        run = self.submit_experiment_run(wait_for_completion=self.wait_for_completion)
        model = run.register_model(
            model_name=self.model_name, model_path=self.model_path
        )
        return model

    def submit_experiment_run(self, wait_for_completion: bool = True):
        """
        Submit run to experiment context

        :param wait_for_completion: should program wait till success before returning
        """
        raise NotImplementedError


class ModelTrainingContext(ModelManagementContext, ABC):
    """
    Interface for Model Management Contexts that Handle Model Training
    """


class LocalTrainingContext(ModelTrainingContext):
    """
    Model Training Context used to run training locally.
    """

    def __init__(
        self,
        subscription_id,
        resource_group,
        workspace_name, configuration_file: str = project_configuration_file,
        train_py=train_py_default,
    ):
        super().__init__(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            run_configuration=get_local_run_configuration(),
            configuration_file=configuration_file,
            train_py=train_py,
        )
        self.args = None

    def submit_experiment_run(self, wait_for_completion=True) -> Run:
        """

        :param wait_for_completion:
        :return:
        """
        assert self.source_directory
        assert self.train_py
        assert self.run_configuration
        assert self.experiment_name
        assert os.path.isfile(self.source_directory + "/" + self.train_py), (
            f"The file {self.train_py} could not be found at "
            f"{self.source_directory}"
        )

        src = ScriptRunConfig(
            source_directory=self.source_directory,
            script=self.train_py,
            arguments=self.args,
            run_config=get_local_run_configuration(),
        )
        self.image_tags["train_py_hash"] = self._get_file_md5(
            self.source_directory + "/" + self.train_py
        )
        exp = Experiment(workspace=self, name=self.experiment_name)
        run = exp.submit(src)
        if wait_for_completion:
            try:
                run.wait_for_completion(show_output=self.show_output)
            except ActivityFailedException as e:
                print(run.get_details())
                raise e
        return run

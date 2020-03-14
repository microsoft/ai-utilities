import copy
import os
from subprocess import PIPE, CalledProcessError

from azureml._base_sdk_common.common import check_valid_resource_name
from azureml._logging import ChainedIdentity
from azureml._restclient import RestClient
from azureml._restclient.clientbase import PAGINATED_KEY
from azureml._restclient.constants import RUN_ORIGIN
from azureml._restclient.exceptions import ServiceException
from azureml._restclient.models import ErrorResponseException
from azureml._restclient.service_context import ServiceContext
from azureml._restclient.workspace_client import WorkspaceClient
from azureml._run_impl.run_history_facade import RunHistoryFacade
from azureml.core import (
    Webservice,
    ComputeTarget,
    Run,
    ScriptRunConfig,
    Experiment,
    Model,
)
from azureml.core.authentication import AbstractAuthentication
from azureml.exceptions import ActivityFailedException
from msrest.exceptions import HttpOperationError

from azure_utils.configuration.notebook_config import project_configuration_file
from azure_utils.configuration.project_configuration import ProjectConfiguration
from azure_utils.machine_learning.contexts.realtime_score_context import (
    MLRealtimeScore,
    DeepRealtimeScore,
)


class MockMLRealtimeScore(MLRealtimeScore):
    def __init__(
        self,
        subscription_id,
        resource_group,
        workspace_name,
        configuration_file,
        score_py,
        train_py,
        model_name="mock_model",
        **kwargs,
    ):
        super().__init__(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            configuration_file=configuration_file,
            train_py=train_py,
            score_py=score_py,
            **kwargs,
        )
        self._subscription_id = subscription_id
        self._resource_group = resource_group
        self._workspace_name = workspace_name
        self.project_configuration_file = configuration_file
        self.score_py = score_py
        self.train_py = train_py
        self.model_name = model_name

        self._auth = MockAuthentication()

        self._service_context = MockServiceContext(
            self._subscription_id,
            self._resource_group,
            self._workspace_name,
            "1",
            self._auth,
        )
        self.wait_for_completion = True

        self.source_directory = "./script"
        self.script = "create_model.py"
        self.create_model_script_file()

    @classmethod
    def get_or_create_workspace(
        cls, configuration_file: str = project_configuration_file, **kwargs
    ):
        project_configuration = ProjectConfiguration(configuration_file)
        assert project_configuration.has_value("subscription_id")
        assert project_configuration.has_value("resource_group")
        assert project_configuration.has_value("workspace_name")
        assert project_configuration.has_value("workspace_region")

        return MockMLRealtimeScore(
            project_configuration.get_value("subscription_id"),
            project_configuration.get_value("resource_group"),
            project_configuration.get_value("workspace_name"),
            project_configuration_file,
            **kwargs,
        )

    @property
    def compute_targets(self):
        return {"mock_aks_1": MockComputeTarget(self, "mock_aks")}

    @property
    def webservices(self):
        return {"mock_aks_1": MockWebService(self, "mock_aks")}

    @property
    def models(self):
        """Return a dictionary where the key is model name, and value is a :class:`azureml.core.model.Model` object.

        Raises a :class:`azureml.exceptions.WebserviceException` if there was a problem interacting with
        model management service.

        :return: A dictionary of models.
        :rtype: dict[str, azureml.core.Model]
        :raises: azureml.exceptions.WebserviceException
        """
        return {}

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
            run_config=self.run_configuration,
        )
        self.image_tags["train_py_hash"] = self._get_file_md5(
            self.source_directory + "/" + self.train_py
        )
        exp = MockExperiment(workspace=self, name=self.experiment_name)
        run = exp.submit(src)
        if wait_for_completion:
            try:
                run.wait_for_completion(show_output=self.show_output)
            except ActivityFailedException as e:
                print(run.get_details())
                raise e
        return run

    def get_or_create_model(self):
        """
        Get or Create Model

        :return: Model from Workspace
        """
        assert self.model_name

        if self.model_name in self.models:
            # if get_model(self.model_name).tags['train_py_hash'] == self.get_file_md5(
            #         self.source_directory + "/" + self.script):
            model = Model(self, name=self.model_name)
            model.download("outputs", exist_ok=True)
            return model

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

        Model(self, "mock_model")
        model = run.register_model(
            model_name=self.model_name, model_path=self.model_path
        )
        return model

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
            run_config=self.run_configuration,
        )
        self.image_tags["train_py_hash"] = self._get_file_md5(
            self.source_directory + "/" + self.train_py
        )
        exp = MockExperiment(workspace=self, name=self.experiment_name)
        run = exp.submit(src)
        if wait_for_completion:
            try:
                run.wait_for_completion(show_output=self.show_output)
            except ActivityFailedException as e:
                print(run.get_details())
                raise e
        return run


class MockDeepRealtimeScore(DeepRealtimeScore):
    def __init__(
        self,
        subscription_id,
        resource_group,
        workspace_name,
        configuration_file,
        score_py,
        train_py,
        model_name="mock_model",
        **kwargs,
    ):
        super().__init__(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            configuration_file=configuration_file,
            train_py=train_py,
            score_py=score_py,
            **kwargs,
        )
        self._subscription_id = subscription_id
        self._resource_group = resource_group
        self._workspace_name = workspace_name
        self.project_configuration_file = configuration_file
        self.score_py = score_py
        self.train_py = train_py
        self.model_name = model_name

        self._auth = MockAuthentication()

        self._service_context = MockServiceContext(
            self._subscription_id,
            self._resource_group,
            self._workspace_name,
            "1",
            self._auth,
        )
        self.wait_for_completion = True

    @classmethod
    def get_or_create_workspace(
        cls, configuration_file: str = project_configuration_file, **kwargs
    ):
        project_configuration = ProjectConfiguration(configuration_file)
        assert project_configuration.has_value("subscription_id")
        assert project_configuration.has_value("resource_group")
        assert project_configuration.has_value("workspace_name")
        assert project_configuration.has_value("workspace_region")

        return MockMLRealtimeScore(
            project_configuration.get_value("subscription_id"),
            project_configuration.get_value("resource_group"),
            project_configuration.get_value("workspace_name"),
            project_configuration_file,
            **kwargs,
        )

    @property
    def compute_targets(self):
        return {"mock_aks_1": MockComputeTarget(self, "mock_aks")}

    @property
    def webservices(self):
        return {"mock_aks_1": MockWebService(self, "mock_aks")}

    @property
    def models(self):
        """Return a dictionary where the key is model name, and value is a :class:`azureml.core.model.Model` object.

        Raises a :class:`azureml.exceptions.WebserviceException` if there was a problem interacting with
        model management service.

        :return: A dictionary of models.
        :rtype: dict[str, azureml.core.Model]
        :raises: azureml.exceptions.WebserviceException
        """
        return {}

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
            run_config=self.run_configuration,
        )
        self.image_tags["train_py_hash"] = self._get_file_md5(
            self.source_directory + "/" + self.train_py
        )
        exp = MockExperiment(workspace=self, name=self.experiment_name)
        run = exp.submit(src)
        if wait_for_completion:
            try:
                run.wait_for_completion(show_output=self.show_output)
            except ActivityFailedException as e:
                print(run.get_details())
                raise e
        return run

    def get_or_create_model(self):
        """
        Get or Create Model

        :return: Model from Workspace
        """
        assert self.model_name

        if self.model_name in self.models:
            # if get_model(self.model_name).tags['train_py_hash'] == self.get_file_md5(
            #         self.source_directory + "/" + self.script):
            model = Model(self, name=self.model_name)
            model.download("outputs", exist_ok=True)
            return model

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

        Model(self, "mock_model")
        model = run.register_model(
            model_name=self.model_name, model_path=self.model_path
        )
        return model

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
            run_config=self.run_configuration,
        )
        self.image_tags["train_py_hash"] = self._get_file_md5(
            self.source_directory + "/" + self.train_py
        )
        exp = MockExperiment(workspace=self, name=self.experiment_name)
        run = exp.submit(src)
        if wait_for_completion:
            try:
                run.wait_for_completion(show_output=self.show_output)
            except ActivityFailedException as e:
                print(run.get_details())
                raise e
        return run


class MockWorkspaceClient(WorkspaceClient):
    def __init__(self, service_context, host=None, **kwargs):
        """
        Constructor of the class.
        """
        self._service_context = service_context
        self._override_host = host
        self._workspace_arguments = [
            self._service_context.subscription_id,
            self._service_context.resource_group_name,
            self._service_context.workspace_name,
        ]
        # super(WorkspaceClient, self).__init__(**kwargs)

        self._custom_headers = {}
        self._client = MockRestClient(service_context._authentication)

    def get_rest_client(self, user_agent=None):
        """get service rest client"""
        return self._service_context._get_run_history_restclient(
            host=self._override_host, user_agent=user_agent
        )

    def get_or_create_experiment(self, experiment_name, is_async=False):
        """
        get or create an experiment by name
        :param experiment_name: experiment name (required)
        :type experiment_name: str
        :param is_async: execute request asynchronously
        :type is_async: bool
        :return:
            If is_async parameter is True,
            the request is called asynchronously.
            The method returns azureml._async_task.AsyncTask object
            If parameter is_async is False or missing,
            return: ~_restclient.models.ExperimentDto
        """

        # Client Create, Get on Conflict
        try:
            return self._execute_with_workspace_arguments(
                self._client.experiment.create,
                experiment_name=experiment_name,
                is_async=is_async,
            )
        except HttpOperationError as e:
            if e.response.status_code == 409:
                experiment = self._execute_with_workspace_arguments(
                    self._client.experiment.get,
                    experiment_name=experiment_name,
                    is_async=is_async,
                )
                if experiment is None:  # This should never happen
                    raise ServiceException(
                        "Failed to get an existing experiment with name "
                        + experiment_name
                    )
            else:
                raise ServiceException(e)

    def _execute_with_workspace_arguments(self, func, *args, **kwargs):
        return self._execute_with_arguments(
            func, copy.deepcopy(self._workspace_arguments), *args, **kwargs
        )

    def _execute_with_arguments(self, func, args_list, *args, **kwargs):
        if not callable(func):
            raise TypeError("Argument is not callable")

        if self._custom_headers:
            kwargs["custom_headers"] = self._custom_headers

        if args:
            args_list.extend(args)
        is_paginated = kwargs.pop(PAGINATED_KEY, False)
        try:
            exec(open("script/create_model.py").read())
        except ErrorResponseException as e:
            raise ServiceException(e)


class MockExperiment(Experiment):
    def __init__(
        self,
        workspace,
        name,
        _skip_name_validation=False,
        _id=None,
        _archived_time=None,
        _create_in_cloud=False,
        _experiment_dto=None,
        **kwargs,
    ):
        """Experiment constructor.

        :param workspace: The workspace object containing the experiment.
        :type workspace: azureml.core.workspace.Workspace
        :param name: The experiment name.
        :type name: str
        :param kwargs: dictionary of keyword args
        :type kwargs: dict
        """
        self._workspace = workspace
        self._name = name
        self._workspace_client = MockWorkspaceClient(workspace.service_context)
        self.train_py = workspace.train_py

        _ident = kwargs.pop(
            "_ident", ChainedIdentity.DELIM.join([self.__class__.__name__, self._name])
        )

        if not _skip_name_validation:
            check_valid_resource_name(name, "Experiment")

        # Get or create the experiment from the workspace
        if _create_in_cloud:
            experiment = self._workspace_client.get_or_create_experiment(
                experiment_name=name
            )
            self._id = experiment.experiment_id
            self._archived_time = experiment.archived_time
            self._extract_from_dto(experiment)
        else:
            self._id = _id
            self._archived_time = _archived_time
            self._extract_from_dto(_experiment_dto)

        # super(Experiment, self).__init__(experiment=self, _ident=_ident, **kwargs)

    def submit(self, config, tags=None, **kwargs):
        """Submit an experiment and return the active created run.

        .. remarks::

            Submit is an asynchronous call to the Azure Machine Learning platform to execute a trial on local
            or remote hardware.  Depending on the configuration, submit will automatically prepare
            your execution environments, execute your code, and capture your source code and results
            into the experiment's run history.

            To submit an experiment you first need to create a configuration object describing
            how the experiment is to be run.  The configuration depends on the type of trial required.

            An example of how to submit an experiment from your local machine is as follows:

            .. code-block:: python

                from azureml.core import ScriptRunConfig

                # run a trial from the train.py code in your current directory
                config = ScriptRunConfig(source_directory='.', script='train.py',
                    run_config=RunConfiguration())
                run = experiment.submit(config)

                # get the url to view the progress of the experiment and then wait
                # until the trial is complete
                print(run.get_portal_url())
                run.wait_for_completion()

            For details on how to configure a run, see the configuration type details.

            * :class:`azureml.core.ScriptRunConfig`
            * :class:`azureml.train.automl.automlconfig.AutoMLConfig`
            * :class:`azureml.pipeline.core.Pipeline`
            * :class:`azureml.pipeline.core.PublishedPipeline`
            * :class:`azureml.pipeline.core.PipelineEndpoint`

            .. note::

                When you submit the training run, a snapshot of the directory that contains your training scripts
                is created and sent to the compute target. It is also stored as part of the experiment in your
                workspace. If you change files and submit the run again, only the changed files will be uploaded.

                To prevent files from being included in the snapshot, create a
                `.gitignore <https://git-scm.com/docs/gitignore>`_
                or `.amlignore` file in the directory and add the
                files to it. The `.amlignore` file uses the same syntax and patterns as the
                `.gitignore <https://git-scm.com/docs/gitignore>`_ file. If both files exist, the `.amlignore` file
                takes precedence.

                For more information, see `Snapshots
                <https://docs.microsoft.com/azure/machine-learning/service/concept-azure-machine-learning-architecture#snapshots>`_

        :param config: The config to be submitted
        :type config: object
        :param tags: Tags to be added to the submitted run, {"tag": "value"}
        :type tags: dict
        :param kwargs: Additional parameters used in submit function for configurations
        :type kwargs: dict
        :return: run
        :rtype: azureml.core.Run
        """
        # Warn user if trying to run GPU image on a local machine
        run = MockRun(self, "1", train_py=self.train_py)
        run.run()
        return run


class MockRunHistoryFacade(RunHistoryFacade):
    def __init__(self, experiment, run_id, origin, run, run_dto=None, **kwargs):
        self._experiment = experiment
        self._origin = origin
        self._run_id = run_id
        self.run = run

        self.run_dto = run_dto if run_dto is not None else self.run.get_run()


class TrainScriptError(Exception):
    pass


class MockRun(Run):
    def __init__(self, experiment, run_id, train_py, **kwargs):
        self._experiment = experiment
        self._run_id = run_id
        self.train_py = train_py

        self._client = MockRunHistoryFacade(
            self._experiment, self._run_id, RUN_ORIGIN, self
        )

        self._runtype = {"azureml.runsource": None}

    def run(self):
        import subprocess

        try:
            subprocess.run(
                ["python", "script/" + self.train_py], stderr=PIPE, check=True
            )
        except CalledProcessError as e:
            raise TrainScriptError(e.stderr.decode("ascii"))

    def wait_for_completion(
        self, show_output=False, wait_post_processing=False, raise_on_error=True
    ):
        return True

    def get_run(self):
        return self

    @property
    def _run_dto(self):
        """Return the internal representation of a run."""
        return {}
        # run_dto = self._client.run_dto
        #
        #
        # if isinstance(run_dto, dict):
        #     self._logger.debug("Return run dto as existing dict")
        #     return run_dto
        # else:
        #     return self._client.run.dto_to_dictionary(run_dto)

    @staticmethod
    def dto_to_dictionary(dto, keep_readonly=True, key_transformer=None):
        """Return a dict that can be JSONify using json.dump.
        :param ~_restclient.models dto: object to transform
        :param bool keep_readonly: If you want to serialize the readonly attributes
        :param function key_transformer: A key transformer function.
                                         Example: attribute_transformer() in msrest.serialization.
        :returns: A dict JSON compatible object
        :rtype: dict
        """
        if dto is None:
            return None

        if not isinstance(dto, Model):
            raise TypeError("Argument is not a Model type")

        if key_transformer is not None:
            return dto.as_dict(
                keep_readonly=keep_readonly, key_transformer=key_transformer
            )

        return dto.as_dict(keep_readonly=keep_readonly)

    def register_model(
        self,
        model_name,
        model_path=None,
        tags=None,
        properties=None,
        model_framework=None,
        model_framework_version=None,
        description=None,
        datasets=None,
        sample_input_dataset=None,
        sample_output_dataset=None,
        resource_configuration=None,
        **kwargs,
    ):
        return Model(self._experiment.workspace, "mock_model")


class MockWebService(Webservice):
    @staticmethod
    def _deploy(
        workspace, name, image, deployment_config, deployment_target, overwrite=False
    ):
        pass

    def run(self, input):
        pass

    def get_token(self):
        pass

    def update(self, *args):
        pass

    def __new__(cls, workspace, name):
        return super(Webservice, cls).__new__(cls)


class MockServiceContext(ServiceContext):
    def __init__(
        self,
        subscription_id,
        resource_group_name,
        workspace_name,
        workspace_id,
        authentication,
    ):
        self._sub_id = subscription_id
        self._rg_name = resource_group_name
        self._ws_name = workspace_name
        self._workspace_id = workspace_id
        self._authentication = authentication

        self.runhistory_restclient = None
        self.artifacts_restclient = None
        self.assets_restclient = None
        self.metrics_restclient = None
        self.project_content_restclient = None
        self.execution_restclient = None
        self.environment_restclient = None
        self.credential_restclient = None
        self.jasmine_restclient = None
        self._session = None

    def _get_run_history_restclient(self, host=None, user_agent=None):
        if self.runhistory_restclient is None:
            host = host if host is not None else self._get_run_history_url()
            self.runhistory_restclient = RestClient(self.get_auth(), base_url=host)
            self._add_user_agent(self.runhistory_restclient, user_agent=user_agent)

        return self.runhistory_restclient

    def _get_run_history_url(self):
        """
        Return the url to the run history service.

        :return: The run history service endpoint.
        :rtype: str
        """
        return "run_history"


class MockAuthentication(AbstractAuthentication):
    def _get_arm_token(self):
        return "123"

    def _get_graph_token(self):
        return "123"

    def _get_all_subscription_ids(self):
        return "123"


class MockComputeTarget(ComputeTarget):
    def _initialize(
        self,
        compute_resource_id,
        name,
        location,
        compute_type,
        tags,
        description,
        created_on,
        modified_on,
        provisioning_state,
        provisioning_errors,
        cluster_resource_id,
        cluster_location,
        workspace,
        mlc_endpoint,
        operation_endpoint,
        auth,
        is_attached,
    ):
        pass

    def refresh_state(self):
        pass

    def delete(self):
        pass

    def detach(self):
        pass

    def serialize(self):
        pass

    @staticmethod
    def deserialize(workspace, object_dict):
        pass

    @staticmethod
    def _validate_get_payload(payload):
        pass

    def __new__(cls, workspace, name):
        return super(ComputeTarget, cls).__new__(cls)


class MockRestClient(RestClient):
    pass

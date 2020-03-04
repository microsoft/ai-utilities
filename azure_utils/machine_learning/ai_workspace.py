"""
AI-Utilities - ai_workspace.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import hashlib
import json
import os
import time

from azure.mgmt.deploymentmanager.models import DeploymentMode
from azure.mgmt.resource import ResourceManagementClient
from azureml.contrib.functions import HTTP_TRIGGER, package
from azureml.core import Workspace, Model, Webservice, ComputeTarget, ScriptRunConfig, Experiment, Image, Environment
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.compute import AksCompute
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.image import ContainerImage
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AksWebservice
from azureml.exceptions import ActivityFailedException

from azure_utils import directory
from azure_utils.configuration.notebook_config import project_configuration_file
from azure_utils.configuration.project_configuration import ProjectConfiguration
from azure_utils.machine_learning.datasets.stack_overflow_data import download_datasets, clean_data, split_duplicates, \
    save_data
from azure_utils.machine_learning.deep.create_deep_model import get_or_create_resnet_image
from azure_utils.machine_learning.realtime.image import get_or_create_lightgbm_image
from azure_utils.machine_learning.train_local import get_local_run_configuration
from azure_utils.machine_learning.utils import get_or_create_workspace_from_project


class AILabWorkspace(Workspace):
    """ AI Workspace """
    image_settings_name = "image_name"
    settings_aks_name = "aks_name"
    settings_aks_service_name = "aks_service_name"

    def __init__(self, subscription_id, resource_group, workspace_name, project_configuration,
                 train_py="train.py", score_py="score.py",
                 run_configuration=get_local_run_configuration(), **kwargs):
        super().__init__(subscription_id, resource_group, workspace_name, **kwargs)
        self.project_configuration = project_configuration

        self.train_py = train_py
        self.score_py = score_py

        self.setting_image_name = "image_name"
        self.settings_aks_name = "aks_name"
        self.setting_service_name = "aks_service_name"
        self.image_tags = None
        self.dockerfile = None
        self.image_dependencies = None
        self.image_description = None
        self.conda_file = None
        self.image_enable_gpu = False
        self.wait_for_completion = True

        self.get_image = None
        self.aks_vm_size: str = "Standard_D4_v2"

        self.model_tags = None
        self.model_description = None
        self.model_name = None
        self.model_path = None

        self.show_output = True

        self.source_directory = "./script"
        self.num_estimators = "1"
        self.args = None
        self.run_configuration = run_configuration
        self.experiment_name = None

        self.node_count = 4
        self.num_replicas: int = 2
        self.cpu_cores: int = 1

        self.get_details()
        self.conda_pack = None
        self.requirements = None
        self.conda_env = CondaDependencies.create(conda_packages=self.conda_pack, pip_packages=self.requirements)

    @classmethod
    def get_or_or_create(cls, configuration_file: str = project_configuration_file,
                         train_py="train.py", score_py="score.py"):
        """ Get or Create Real-time Endpoint

        :param configuration_file:
        :param train_py:
        :param score_py:
        :param kwargs: keyword args
        """
        workspace = cls.get_or_create_workspace(configuration_file, train_py=train_py, score_py=score_py)
        model = workspace.get_or_create_model()
        config = workspace.get_or_create_image_configuration()
        workspace.get_or_create_image(config, models=[model])
        aks_target = workspace.get_or_create_aks()
        web_service = workspace.get_or_create_aks_service(aks_target)
        return workspace, web_service

    @classmethod
    def get_or_or_create_2(cls, configuration_file: str = project_configuration_file,
                           train_py="train.py", score_py="score.py"):
        """ Get or Create Real-time Endpoint

        :param configuration_file:
        :param train_py:
        :param score_py:
        :param kwargs: keyword args
        """
        workspace = cls.get_or_create_workspace(configuration_file, train_py=train_py, score_py=score_py)
        model = workspace.get_or_create_model()
        inference_config = workspace.get_inference_config()
        aks_target = workspace.get_or_create_aks()
        web_service = workspace.get_or_create_aks_service_2(model, aks_target, inference_config)

        return workspace, web_service

    def get_or_create_aks_service_2(self, model, aks_target, inference_config):
        assert self.project_configuration.has_value(self.settings_aks_service_name)
        aks_service_name = self.project_configuration.get_value(self.settings_aks_service_name)

        if self.has_web_service(aks_service_name) and self.get_web_service_state(aks_service_name) != "Failed":
            self.create_kube_config(aks_target)
            return self.get_web_service(aks_service_name)

        aks_service = Model.deploy(self, aks_service_name, models=[model], inference_config=inference_config,
                                   deployment_target=aks_target)
        if self.wait_for_completion:
            aks_service.wait_for_deployment(show_output=self.show_output)
            configure_ping_test("ping-test-" + aks_service_name, self.get_details()['applicationInsights'],
                                aks_service.scoring_uri, aks_service.get_keys()[0])
        return aks_service

    @classmethod
    def get_or_create_workspace(cls, configuration_file: str = project_configuration_file,
                                train_py="train.py", score_py="score.py"):
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
                 project_configuration, train_py=train_py, score_py=score_py)
        return ws

    def get_or_create_model(self) -> Model:
        """
        Get or Create Model

        :param kwargs: keyword args
        :return: Model from Workspace
        """
        assert self.model_name

        if self.model_name in self.models:
            # if get_model(self.model_name).tags['train_py_hash'] == self.get_file_md5(
            #         self.source_directory + "/" + self.script):
            model = Model(self, name=self.model_name)
            model.download("outputs", exist_ok=True)
            return model

        run = self.submit_experiment_run(wait_for_completion=self.wait_for_completion)
        model = run.register_model(model_name=self.model_name, model_path=self.model_path)

        assert model
        if self.show_output:
            print(model.name, model.version, model.url, sep="\n")
        return model

    def get_or_create_aks(self):
        """
        Get or Create AKS Cluster

        :param show_output:
        :param node_count:
        :param vm_size:
        :param configuration_file:
        :param kwargs: keyword args
        :return: AKS Compute from Workspace
        """
        assert self.project_configuration.has_value(self.settings_aks_name)
        assert self.project_configuration.has_value(self.settings_aks_service_name)
        assert "_" not in self.project_configuration.get_value(
            self.settings_aks_service_name), self.settings_aks_service_name + " can not contain _"
        assert self.project_configuration.has_value("workspace_region")

        aks_name = self.project_configuration.get_value(self.settings_aks_name)
        aks_service_name = self.project_configuration.get_value(self.settings_aks_service_name)
        aks_location = self.project_configuration.get_value("workspace_region")

        workspace_compute = self.compute_targets
        if aks_name in workspace_compute:
            return workspace_compute[aks_name]

        prov_config = AksCompute.provisioning_configuration(agent_count=self.node_count, vm_size=self.aks_vm_size,
                                                            location=aks_location)

        deploy_aks_start = time.time()
        aks_target = ComputeTarget.create(workspace=self, name=aks_name, provisioning_configuration=prov_config)

        aks_target.wait_for_completion(show_output=True)
        if self.show_output:
            deployment_time_secs = str(time.time() - deploy_aks_start)
            print("Deployed AKS with name "
                  + aks_service_name + ". Took " + deployment_time_secs + " seconds.")
            print(aks_target.provisioning_state)
            print(aks_target.provisioning_errors)
        aks_status = aks_target.get_status()
        assert aks_status == 'Succeeded', 'AKS failed to create'
        return aks_target

    def get_or_create_aks_service(self, aks_target) -> AksWebservice:
        """
        Get or Create AKS Service with new or existing Kubernetes Compute

        :param show_output: toggle on/off standard output. default: `True`
        :return: New or Existing Kubernetes Web Service
        """
        assert self.aks_vm_size

        assert self.project_configuration.has_value(self.setting_service_name)
        assert self.project_configuration.has_value(self.setting_image_name)

        aks_service_name = self.project_configuration.get_value(self.setting_service_name)
        image_name = self.project_configuration.get_value(self.setting_image_name)

        if self.has_web_service(aks_service_name) and self.get_web_service_state(aks_service_name) != "Failed":
            self.create_kube_config(aks_target)
            return self.get_web_service(aks_service_name)

        aks_config = self.get_aks_deployment_config()

        if image_name not in self.images:
            self.get_image()
        image = self.images[image_name]

        deploy_from_image_start = time.time()

        aks_service = Webservice.deploy_from_image(workspace=self, name=aks_service_name, image=image,
                                                   deployment_config=aks_config, deployment_target=aks_target,
                                                   overwrite=True)
        try:
            aks_service.wait_for_deployment(show_output=self.show_output)

            configure_ping_test("ping-test-" + aks_service_name, self.get_details()['applicationInsights'],
                                aks_service.scoring_uri, aks_service.get_keys()[0])
        except:
            print(aks_service.get_logs())
            raise Exception
        if self.show_output:
            deployment_time_secs = str(time.time() - deploy_from_image_start)
            print("Deployed Image with name "
                  + aks_service_name + ". Took " + deployment_time_secs + " seconds.")
            print(aks_service.state)
            print(aks_service.get_logs())

        return aks_service

    def get_aks_deployment_config(self):
        aks_config = AksWebservice.deploy_configuration(num_replicas=self.num_replicas, cpu_cores=self.cpu_cores,
                                                        enable_app_insights=True, collect_model_data=True)
        return aks_config

    def get_or_create_image_configuration(self, **kwargs):
        """ Get or Create new Docker Image Configuration for Machine Learning Workspace

        :param kwargs: keyword args
        """
        """
        Image Configuration for running LightGBM in Azure Machine Learning Workspace

        :return: new image configuration for Machine Learning Workspace
        """
        assert self.score_py
        assert self.conda_file
        assert self.image_description
        assert self.image_tags

        self.write_conda_env()
        assert os.path.isfile(self.conda_file)

        self.image_tags['score_py_hash'] = self._get_file_md5(self.score_py)
        return ContainerImage.image_configuration(execution_script=self.score_py, runtime="python",
                                                  conda_file=self.conda_file, description=self.image_description,
                                                  dependencies=self.image_dependencies, docker_file=self.dockerfile,
                                                  tags=self.image_tags, enable_gpu=self.image_enable_gpu, **kwargs)

    def get_or_create_image(self, image_config, models=None):
        """Get or Create new Docker Image from Machine Learning Workspace

        :param image_config:
        :param image_settings_name:
        :param models:
        :param kwargs: keyword args
        """
        if not models:
            models = []

        assert self.project_configuration.has_value(self.setting_image_name)
        image_name = self.project_configuration.get_value(self.setting_image_name)

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
            deployment_time_secs = str(time.time() - image_create_start)
            print("Deployed Image with name " + image_name + ". Took " + deployment_time_secs + " seconds.")
            print(image.name)
            print(image.version)
            print(image.image_build_log_uri)
        return image

    @classmethod
    def get_or_or_create_function_endpoint(cls):
        """ Get or Create Real-time Endpoint

        :param kwargs: keyword args
        """
        workspace = cls.get_or_create_workspace()
        model = workspace.get_or_create_model()
        config = workspace.get_or_create_function_image_configuration()
        image = workspace.get_or_create_function_image(config, models=[model])
        return workspace, image

    def get_or_create_function_image_configuration(self):
        """ Get or Create new Docker Image Configuration for Machine Learning Workspace

        :param kwargs: keyword args
        """
        """
        Image Configuration for running LightGBM in Azure Machine Learning Workspace

        :return: new image configuration for Machine Learning Workspace
        """
        assert self.score_py
        assert self.conda_file
        assert self.image_description
        assert self.image_tags

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

    def get_or_create_function_image(self, config, models):
        blob = package(self, models, config, functions_enabled=True, trigger=HTTP_TRIGGER)
        blob.wait_for_creation(show_output=True)
        # Display the package location/ACR path
        print(blob.location)

    @staticmethod
    def _get_file_md5(file_name):
        hasher = hashlib.md5()
        with open(file_name, 'rb') as afile:
            buf = afile.read()
            hasher.update(buf)
        file_hash = hasher.hexdigest()
        return file_hash

    def has_web_service(self, service_name):
        assert self.webservices
        return service_name in self.webservices

    def get_web_service_state(self, service_name):
        web_service = self.get_web_service(service_name)
        web_service.update_deployment_state()
        assert web_service.state
        return web_service.state

    def get_web_service(self, service_name):
        assert self.webservices[service_name]
        return self.webservices[service_name]

    def create_kube_config(self, aks_target: AksCompute):
        os.makedirs(os.path.join(os.path.expanduser('~'), '.kube'), exist_ok=True)
        config_path = os.path.join(os.path.expanduser('~'), '.kube/config')
        with open(config_path, 'a') as f:
            f.write(aks_target.get_credentials()['userKubeConfig'])

    def submit_experiment_run(self, wait_for_completion=True):
        assert self.source_directory
        assert self.train_py
        assert self.args
        assert self.run_configuration
        assert self.experiment_name

        src = ScriptRunConfig(
            source_directory=self.source_directory,
            script=self.train_py,
            arguments=self.args,
            run_config=self.run_configuration,
        )
        self.image_tags['train_py_hash'] = self._get_file_md5(self.source_directory + "/" + self.train_py)
        exp = Experiment(workspace=self, name=self.experiment_name)
        run = exp.submit(src)
        if wait_for_completion:
            try:
                run.wait_for_completion(show_output=self.show_output)
            except ActivityFailedException as e:
                print(run.get_details())
                raise e
        return run

    def test_service_local(self):
        Model(self, self.model_name).download(exist_ok=True)
        exec(open(self.score_py).read())
        exec("init()")
        exec("response = run(MockRequest())")
        exec("assert response")

    def get_inference_config(self):
        environment = Environment("conda-env")
        environment.python.conda_dependencies = self.conda_env

        inference_config = InferenceConfig(entry_script=self.score_py, environment=environment)
        return inference_config

    def write_conda_env(self):
        with open(self.conda_file, "w") as file:
            file.write(self.conda_env.serialize_to_string())


class MLRealtimeScore(AILabWorkspace):
    """ Light GBM Real Time Scoring"""

    def __init__(self, subscription_id, resource_group, workspace_name, project_configuration,
                 run_configuration=get_local_run_configuration(), **kwargs):
        super().__init__(subscription_id, resource_group, workspace_name, project_configuration, **kwargs)
        # self.write_conda_env()
        self.get_docker_file()

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

        self.description = "Image with lightgbm model"
        self.tags = {"area": "text", "type": "lightgbm"}
        self.get_image = get_or_create_lightgbm_image

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
        self.run_configuration = run_configuration

        self.test_size = 0.21
        self.min_text = 150
        self.min_dupes = 12
        self.match = 20

        self.prepare_data()

        self.conda_pack = [
            "scikit-learn==0.19.1",
            "pandas==0.23.3"
        ]
        self.requirements = [
            "lightgbm==2.1.2",
            "azureml-defaults==1.0.57",
            "azureml-contrib-services",
            "opencensus-ext-flask",
            "Microsoft-AI-Azure-Utility-Samples"
        ]
        self.conda_env = CondaDependencies.create(conda_packages=self.conda_pack, pip_packages=self.requirements)

    def get_docker_file(self):
        self.dockerfile = "dockerfile"
        with open(self.dockerfile, "w") as file:
            file.write("RUN apt update -y && apt upgrade -y && apt install -y build-essential")

    def prepare_data(self):
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
                                                                                     questions,
                                                                                     self.show_output, self.test_size)

            save_data(balanced_pairs_test, balanced_pairs_train, dupes_test, dupes_test_path, outputs_path, questions,
                      questions_path, self.show_output)


class DeepRealtimeScore(AILabWorkspace):
    """ Resnet Real-time Scoring"""
    image_settings_name = "mydeepimage"
    settings_aks_name = "deep_aks_name"
    settings_aks_service_name = "deep_aks_service_name"

    def __init__(self, subscription_id, resource_group, workspace_name, project_configuration,
                 train_py="create_deep_model.py", score_py="driver.py", conda_file="img_env.yml",
                 setting_image_name="deep_image_name", settings_aks_name="deep_aks_name",
                 setting_service_name="deep_aks_service_name", **kwargs):
        super().__init__(subscription_id, resource_group, workspace_name, project_configuration,
                         train_py=train_py, score_py=score_py, **kwargs)
        self.setting_image_name = setting_image_name
        self.settings_aks_name = settings_aks_name
        self.setting_service_name = setting_service_name

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
        self.conda_pack = [
            "tensorflow-gpu==1.14.0"
        ]
        self.requirements = [
            "keras==2.2.0",
            "Pillow==5.2.0",
            "azureml-defaults",
            "azureml-contrib-services",
            "toolz==0.9.0"
        ]
        self.conda_env = CondaDependencies.create(conda_packages=self.conda_pack, pip_packages=self.requirements)

        # Image Configuration
        self.get_image = get_or_create_resnet_image
        self.image_tags = {"name": "AKS", "project": "AML"}
        self.image_description = "Image for AKS Deployment Tutorial"
        self.image_dependencies = ["resnet152.py"]
        self.image_enable_gpu = True

        # Kubernetes Configuration
        self.aks_vm_size = "Standard_NC6"


#
#         if not os.path.isfile(execution_script):
#             with open(execution_script, "w") as file:
#                 file.write("""
#         import os.path
#
#         from resnet152 import ResNet152
#         from keras.preprocessing import image
#         from keras.applications.imagenet_utils import preprocess_input, decode_predictions
#         from azureml.core import Workspace
#         from azureml.core.model import Model
#         from azureml.contrib.services.aml_request import rawhttp
#         from azureml.contrib.services.aml_response import AMLResponse
#         from toolz import compose
#         import numpy as np
#         from PIL import Image, ImageOps
#
#         _NUMBER_RESULTS = 3
#
#
#         def init():
#             global model
#             dir = "."
#             if os.getenv('AZUREML_MODEL_DIR'):
#                dir = os.getenv('AZUREML_MODEL_DIR')
#
#             assert os.path.isfile(dir + "/model.pkl")
#
#             from resnet152 import ResNet152
#             model = ResNet152()
#             model.load_weights(dir + "/model.pkl")
#
#         @rawhttp
#         def run(request):
#             if request.method == 'POST':
#                 #preprocess
#                 transform_input = compose(_pil_to_numpy, _image_ref_to_pil_image)
#                 transformed_dict = {key: transform_input(img_ref) for key, img_ref in request.files.items()}
#                 img_array = preprocess_input(np.stack(list(transformed_dict.values())))
#
#                 preds = model.predict(img_array)
#
#                 # Postprocess
#                 # nverting predictions to float64 since we are able to serialize float64 but not float32
#                 preds = decode_predictions(preds.astype(np.float64), top=_NUMBER_RESULTS)
#                 return dict(zip(transformed_dict.keys(), preds))
#
#             if request.method == 'GET':
#                 return {"azEnvironment": "Azure"}
#             return AMLResponse("bad request", 500)
#
#
#         def _image_ref_to_pil_image(image_ref):
#             return Image.open(image_ref).convert("RGB")
#
#
#         def _pil_to_numpy(pil_image):
#             img = ImageOps.fit(pil_image, (224, 224), Image.ANTIALIAS)
#             img = image.img_to_array(img)
#             return img
#
#
#         """)
#
#         if not os.path.isfile("script/" + self.script):
#             os.makedirs("script", exist_ok=True)
#             with open("script/" + self.script, "w") as file:
#                 file.write("""
# import argparse
# import os
#
# from azureml.core import Run
# from sklearn.externals import joblib
#
# from resnet152 import ResNet152
#
# if __name__ == '__main__':
#     resnet_152 = ResNet152(weights="imagenet")
#     resnet_152.save_weights("outputs/model.pkl")
#
# """)


class MockRequest:
    method = 'GET'


def configure_ping_test(ping_test_name, app_name, ping_url, ping_token):
    project_configuration = ProjectConfiguration(project_configuration_file)
    assert project_configuration.has_value('subscription_id')
    credentials = AzureCliAuthentication()
    client = ResourceManagementClient(credentials, project_configuration.get_value('subscription_id'))
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'webtest.json')
    with open(template_path, 'r') as template_file_fd:
        template = json.load(template_file_fd)

    parameters = {
        'appName': app_name.split("components/")[1],
        'pingURL': ping_url,
        'pingToken': ping_token,
        'location': project_configuration.get_value('workspace_region'),
        'pingTestName': ping_test_name + "-" + project_configuration.get_value('workspace_region')
    }
    parameters = {k: {'value': v} for k, v in parameters.items()}

    deployment_properties = {
        'mode': DeploymentMode.incremental,
        'template': template,
        'parameters': parameters
    }

    deployment_async_operation = client.deployments.create_or_update(
        project_configuration.get_value('resource_group'),
        'add-web-test',
        deployment_properties
    )
    deployment_async_operation.wait()

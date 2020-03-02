"""
AI-Utilities - ai_workspace.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import os
import time

import pandas as pd
from azureml.contrib.functions import HTTP_TRIGGER, package
from azureml.core import Workspace, Model, Webservice, ComputeTarget, ScriptRunConfig, Experiment
from azureml.core.compute import AksCompute
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.image import ContainerImage
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AksWebservice

from azure_utils import directory
from azure_utils.configuration.notebook_config import project_configuration_file
from azure_utils.configuration.project_configuration import ProjectConfiguration
from azure_utils.machine_learning.datasets.stack_overflow_data import download_datasets, clean_data, split_duplicates, \
    save_data
from azure_utils.machine_learning.deep.create_deep_model import get_or_create_resnet_image, download_test_image
from azure_utils.machine_learning.model import has_model, get_model
from azure_utils.machine_learning.realtime.image import get_or_create_lightgbm_image
from azure_utils.machine_learning.train_local import get_local_run_configuration
from azure_utils.machine_learning.utils import get_or_create_workspace_from_project, get_workspace_from_config
from resnet152 import ResNet152


class AILabWorkspace(Workspace):
    """ AI Workspace """
    image_settings_name = "image_name"
    settings_aks_name = "aks_name"
    settings_aks_service_name = "aks_service_name"

    def __init__(self, subscription_id, resource_group, workspace_name,
                 run_configuration=get_local_run_configuration()):
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

        self.model_tags = None
        self.model_description = None
        self.model_name = None
        self.model_path = None

        self.show_output = True

        self.script = None
        self.source_directory = None
        self.args = None
        self.run_configuration = run_configuration
        self.experiment_name = None

    @classmethod
    def get_or_or_create_realtime_endpoint(cls):
        """ Get or Create Real-time Endpoint

        :param kwargs: keyword args
        """
        workspace = cls.get_or_create_workspace()
        models = [workspace.get_or_create_model()]
        config = workspace.get_or_create_image_configuration()
        cls.get_or_create_image(config, cls.image_settings_name, models=models)
        return workspace.get_or_create_service()

    @classmethod
    def get_or_or_create_function_endpoint(cls):
        """ Get or Create Real-time Endpoint

        :param kwargs: keyword args
        """
        workspace = cls.get_or_create_workspace()
        models = [workspace.get_or_create_model()]
        config = workspace.get_or_create_function_image_configuration()
        workspace.get_or_create_function_image(config, models=models)

    @classmethod
    def get_or_create_workspace(cls, configuration_file: str = project_configuration_file):
        """ Get or create a workspace if it doesn't exist.

        :param configuration_file:
        """
        project_configuration = ProjectConfiguration(configuration_file)
        assert project_configuration.has_settings('subscription_id')
        assert project_configuration.has_settings('resource_group')
        assert project_configuration.has_settings('workspace_name')

        return cls(project_configuration.get_value('subscription_id'),
                   project_configuration.get_value('resource_group'),
                   project_configuration.get_value('workspace_name'))

    def get_or_create_model(self) -> Model:
        """
        Get or Create Model

        :param kwargs: keyword args
        :return: Model from Workspace
        """
        assert self.model_name

        if has_model(self.model_name):
            return get_model(self.model_name)

        model = self.register_model()
        assert model
        if self.show_output:
            print(model.name, model.version, model.url, sep="\n")
        return model

    def register_model(self, **kwargs) -> Model:
        pass

    def get_or_create_aks(self, configuration_file: str = project_configuration_file, vm_size: str = "Standard_D4_v2",
                          node_count: int = 4, show_output: bool = True):
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

        workspace_compute = self.compute_targets
        if aks_name in workspace_compute:
            return workspace_compute[aks_name]

        prov_config = AksCompute.provisioning_configuration(agent_count=node_count, vm_size=vm_size,
                                                            location=aks_location)

        deploy_aks_start = time.time()
        aks_target = ComputeTarget.create(workspace=self, name=aks_name, provisioning_configuration=prov_config)

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
                                                  tags=self.tags, enable_gpu=self.enable_gpu, **kwargs)

    def get_or_create_function_image_configuration(self):
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

        from azureml.core.environment import Environment
        from azureml.core.conda_dependencies import CondaDependencies

        # Create an environment and add conda dependencies to it
        myenv = Environment(name="myenv")
        # Enable Docker based environment
        myenv.docker.enabled = True
        # Build conda dependencies
        myenv.python.conda_dependencies = CondaDependencies.create(conda_packages=['scikit-learn'],
                                                                   pip_packages=['azureml-defaults'])
        return InferenceConfig(entry_script="score.py", environment=myenv)

    def get_or_create_service(self, configuration_file: str = project_configuration_file,
                              node_count: int = 4, num_replicas: int = 2,
                              cpu_cores: int = 1, show_output: bool = True) -> AksWebservice:
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

        if aks_service_name in self.webservices:
            return self.webservices[aks_service_name]

        aks_config = AksWebservice.deploy_configuration(num_replicas=num_replicas, cpu_cores=cpu_cores)

        if image_name not in self.images:
            self.get_image()
        image = self.images[image_name]

        deploy_from_image_start = time.time()
        aks_service = Webservice.deploy_from_image(workspace=self, name=aks_service_name, image=image,
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
    def get_or_create_image(image_config, image_settings_name, models=None, show_output=True,
                            configuration_file: str = project_configuration_file):
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

    def submit_experiment(self, wait_for_completion=True):
        assert self.source_directory
        assert self.script
        assert self.args
        assert self.run_configuration
        assert self.experiment_name

        src = ScriptRunConfig(
            source_directory=self.source_directory,
            script=self.script,
            arguments=self.args,
            run_config=self.run_configuration,
        )
        run = Experiment(workspace=self, name=self.experiment_name).submit(src)
        if wait_for_completion:
            run.wait_for_completion(show_output=self.show_output)
        return run

    def register_model_from_run(self, run):
        return run.register_model(model_name=self.model_name, model_path=self.model_path)

    def get_or_create_function_image(self, config, models):
        blob = package(self, models, config, functions_enabled=True, trigger=HTTP_TRIGGER)
        blob.wait_for_creation(show_output=True)
        # Display the package location/ACR path
        print(blob.location)


class RTSWorkspace(AILabWorkspace):
    """ Light GBM Real Time Scoring"""

    def __init__(self, subscription_id, resource_group, workspace_name,
                 run_configuration=get_local_run_configuration()):
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
        self.dependencies = None

        self.conda_file = "lgbmenv.yml"
        with open(self.conda_file, "w") as file:
            file.write(lgbmenv.serialize_to_string())

        self.dockerfile = "dockerfile"
        with open(self.dockerfile, "w") as file:
            file.write("RUN apt update -y && apt upgrade -y && apt install -y build-essential")

        self.execution_script = "score.py"
        with open(self.execution_script, 'w') as file:
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

        self.model_name = "question_match_model"
        self.num_estimators = "1"
        self.experiment_name = "mlaks-train-on-local"
        self.model_path = "./outputs/model.pkl"
        self.script = "create_model.py"
        self.source_directory = "./script"
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

    def register_model(self) -> Model:
        if not os.path.isfile("script/create_model.py"):
            os.makedirs("script", exist_ok=True)

            create_model_py = "from azure_utils.machine_learning import create_model\n\nif __name__ == '__main__':\n" \
                              "    create_model.main()"
            with open("script/create_model.py", "w") as file:
                file.write(create_model_py)

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

        run = self.submit_experiment()
        return self.register_model_from_run(run)


class DeepRTSWorkspace(AILabWorkspace):
    """ Resnet Real-time Scoring"""
    image_settings_name = "mydeepimage"
    settings_aks_name = "deep_aks_name"
    settings_aks_service_name = "deep_aks_service_name"

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
        with open(execution_script, "w") as file:
            file.write('\nfrom resnet152 import ResNet152\nfrom keras.preprocessing import image\nfrom '
                       'keras.applications.imagenet_utils import preprocess_input, decode_predictions\nfrom '
                       'azureml.contrib.services.aml_request import rawhttp\nfrom azureml.core.model import '
                       'Model\nfrom azureml.contrib.services.aml_response import AMLResponse\nfrom toolz '
                       'import compose\nimport numpy as np\nimport timeit as t\nfrom PIL import Image, '
                       'ImageOps\nimport logging\n\n_NUMBER_RESULTS = 3\n\n\ndef _image_ref_to_pil_image('
                       'image_ref):\n    """ Load image with PIL (RGB)\n    """\n    return Image.open('
                       'image_ref).convert("RGB")\n\n\ndef _pil_to_numpy(pil_image):\n    img = '
                       'ImageOps.fit(pil_image, (224, 224), Image.ANTIALIAS)\n    img = image.img_to_array('
                       'img)\n    return img\n\n\ndef _create_scoring_func():\n    """ Initialize ResNet '
                       '152 Model\n    """\n    logger = logging.getLogger("model_driver")\n    start = '
                       't.default_timer()\n    model_name = "resnet_model"\n    model_path = '
                       'Model.get_model_path(model_name)\n    model = ResNet152()\n    model.load_weights('
                       'model_path)\n    end = t.default_timer()\n\n    load_time = "Model loading time: '
                       '{0} ms".format(round((end - start) * 1000, 2))\n    logger.info(load_time)\n\n    '
                       'def call_model(img_array_list):\n        img_array = np.stack(img_array_list)\n     '
                       '   img_array = preprocess_input(img_array)\n        preds = model.predict('
                       'img_array)\n        # Converting predictions to float64 since we are able to '
                       'serialize float64 but not float32\n        preds = decode_predictions(preds.astype('
                       'np.float64), top=_NUMBER_RESULTS)\n        return preds\n\n    return '
                       'call_model\n\n\ndef get_model_api():\n    logger = logging.getLogger('
                       '"model_driver")\n    scoring_func = _create_scoring_func()\n\n    '
                       'def process_and_score(images_dict):\n        """ Classify the input using the '
                       'loaded model\n        """\n        start = t.default_timer()\n        logger.info('
                       '"Scoring {} images".format(len(images_dict)))\n        transform_input = compose('
                       '_pil_to_numpy, _image_ref_to_pil_image)\n        transformed_dict = {\n            '
                       'key: transform_input(img_ref) for key, img_ref in images_dict.items()\n        }\n  '
                       '      preds = scoring_func(list(transformed_dict.values()))\n        preds = dict('
                       'zip(transformed_dict.keys(), preds))\n        end = t.default_timer()\n\n        '
                       'logger.info("Predictions: {0}".format(preds))\n        logger.info("Predictions '
                       'took {0} ms".format(round((end - start) * 1000, 2)))\n        return (preds, '
                       '"Computed in {0} ms".format(round((end - start) * 1000, 2)))\n\n    return '
                       'process_and_score\n\n\ndef init():\n    """ Initialise the model and scoring '
                       'function\n    """\n    global process_and_score\n    process_and_score = '
                       'get_model_api()\n\n\n@rawhttp\ndef run(request):\n    """ Make a prediction based '
                       'on the data passed in using the preloaded model\n    """\n    if request.method == '
                       '\'POST\':\n        return process_and_score(request.files)\n    if request.method '
                       '== \'GET\':\n        resp_body = {\n            "azEnvironment": "Azure",'
                       '\n            "location": "westus2",\n            "osType": "Ubuntu 16.04",'
                       '\n            "resourceGroupName": "",\n            "resourceId": "",\n            '
                       '"sku": "",\n            "subscriptionId": "",\n            "uniqueId": '
                       '"PythonMLRST",\n            "vmSize": "",\n            "zone": "",\n            '
                       '"isServer": False,\n            "version": ""\n        }\n        return '
                       'resp_body\n    return AMLResponse("bad request", 500)')
        self.execution_script = execution_script

        self.description = "Image for AKS Deployment Tutorial"
        self.dependencies = ["resnet152.py"]
        self.tags = {"name": "AKS", "project": "AML"}
        self.enable_gpu = True

        self.vm_size = "Standard_NC6"

        self.model_tags = {"model": "dl", "framework": "resnet"}
        self.model_description = "resnet 152 model"
        self.model_name = "resnet_model"
        self.model_path = "model_resnet_weights.h5"

        self.get_image = get_or_create_resnet_image

    def register_model(self) -> Model:
        resnet_152 = ResNet152(weights="imagenet")
        resnet_152.save_weights(self.model_path)
        # Clear GPU memory
        from keras import backend
        backend.clear_session()

        return Model.register(
            model_path=self.model_path,  # this points to a local file
            model_name=self.model_name,  # this is the name the   model is registered as
            tags=self.model_tags,
            description=self.model_description,
            workspace=self,
        )

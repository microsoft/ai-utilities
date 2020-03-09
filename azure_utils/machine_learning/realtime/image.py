"""
AI-Utilities - image.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import time

from azureml.core import Image
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.image import ContainerImage
from azureml.core.image.container import ContainerImageConfig

from azure_utils.configuration.notebook_config import project_configuration_file
from azure_utils.configuration.project_configuration import ProjectConfiguration
from azure_utils.machine_learning.realtime.kubernetes import get_dupes_test
from azure_utils.machine_learning.utils import get_or_create_workspace_from_project
from azure_utils.utilities import text_to_json


def get_or_create_lightgbm_image(
    configuration_file: str = project_configuration_file,
    show_output: bool = True,
    models: list = None,
    dependencies=None,
    image_settings_name="image_name",
) -> ContainerImage:
    """
    Get or Create new Docker Image from Machine Learning Workspace

    :param configuration_file: path to project configuration file. default: project.yml
    :param show_output: toggle on/off standard output. default: `True`
    :param models Name of Model to package with Image from Machine Learning Workspace
    :param dependencies: List of files to include in image
    :param image_settings_name: Setting from Project Configuration
    :return: New or Existing Docker Image for deployment to Kubernetes Compute
    """
    image_config = create_lightgbm_image_config(dependencies=dependencies)

    if not models:
        models = []

    return get_or_create_image(
        image_config, image_settings_name, show_output, models, configuration_file
    )


def get_or_create_image(
    image_config,
    image_settings_name,
    show_output,
    models=None,
    configuration_file: str = project_configuration_file,
):
    """

    :param image_config:
    :param image_settings_name:
    :param models:
    :param show_output:
    :param configuration_file: path to project configuration file. default: project.yml
:return:
    """
    if not models:
        models = []

    project_configuration = ProjectConfiguration(configuration_file)

    assert project_configuration.has_value(image_settings_name)
    image_name = project_configuration.get_value(image_settings_name)

    workspace = get_or_create_workspace_from_project(
        project_configuration, show_output=show_output
    )

    workspace_images = workspace.images
    if (
        image_name in workspace_images
        and workspace_images[image_name].creation_state != "Failed"
    ):
        return workspace_images[image_name]

    image_create_start = time.time()
    image = ContainerImage.create(
        name=image_name, models=models, image_config=image_config, workspace=workspace
    )
    image.wait_for_creation(show_output=show_output)
    assert image.creation_state != "Failed"
    if show_output:
        print_image_deployment_info(image, image_name, image_create_start)
    return image


def print_deployment_time(service_name: str, deploy_start_time: float, service_id: str):
    """
    Print the deployment time of the service so it can be captured in devops logs.

    :param service_name:
    :param deploy_start_time:
    :param service_id:
    """
    deployment_time_secs = str(time.time() - deploy_start_time)
    print(
        f"Deployed {service_id} with name {service_name}. Took {deployment_time_secs} seconds."
    )


def print_image_deployment_info(
    image: Image, image_name: str, image_create_start: float
):
    """
    Print general information about deploying an image.

    :param image:
    :param image_name:
    :param image_create_start:
    """
    print_deployment_time(image_name, image_create_start, "Image")
    print(image.name)
    print(image.version)
    print(image.image_build_log_uri)


def create_lightgbm_image_config(
    conda_file="lgbmenv.yml", execution_script="score.py", dependencies=None
) -> ContainerImageConfig:
    """
    Image Configuration for running LightGBM in Azure Machine Learning Workspace

    :param conda_file: file name of LightGBM Conda Env File. This file is created if it does not exist.
     default: lgbmenv.yml
    :param execution_script: webservice file. default: score.py
    :param dependencies: Files required for image.
    :return: new image configuration for Machine Learning Workspace
    """
    create_lightgbm_conda_file(conda_file)

    dockerfile = "dockerfile"
    with open(dockerfile, "w") as file:
        file.write(
            "RUN apt update -y && apt upgrade -y && apt install -y build-essential"
        )

    with open("score.py", "w") as file:
        file.write(
            """        
import json
import logging


def init():
    logger = logging.getLogger("scoring_script")
    logger.info("init")


def run():
    logger = logging.getLogger("scoring_script")
    logger.info("run")
    return json.dumps({'call': True})
"""
        )
    description = "Image with lightgbm model"
    tags = {"area": "text", "type": "lightgbm"}
    return ContainerImage.image_configuration(
        execution_script=execution_script,
        runtime="python",
        conda_file=conda_file,
        description=description,
        dependencies=dependencies,
        docker_file=dockerfile,
        tags=tags,
    )


def create_lightgbm_conda_file(conda_file: str = "lgbmenv.yml"):
    """
    Create new Conda File with LightGBM requirements.

    :param conda_file: filename of LightGBM conda file, which is created during call.
    """
    conda_pack = ["scikit-learn==0.19.1", "pandas==0.23.3"]
    requirements = [
        "lightgbm==2.1.2",
        "azureml-defaults==1.0.57",
        "azureml-contrib-services",
        "Microsoft-AI-Azure-Utility-Samples",
    ]
    lgbmenv = CondaDependencies.create(
        conda_packages=conda_pack, pip_packages=requirements
    )
    with open(conda_file, "w") as file:
        file.write(lgbmenv.serialize_to_string())


def lightgbm_test_image_locally(image: Image, directory: str):
    """
    Test LightGBM image Locally.

    :param image: Machine Learning Image to test.
    :param directory: root directory that contains data directory.
    """
    dupes_test = get_dupes_test(directory)
    text_to_score = dupes_test.iloc[0, 4]
    json_text = text_to_json(text_to_score)
    image.run(input_data=json_text)

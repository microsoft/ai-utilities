"""
AI-Utilities - image.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import time

import pandas as pd
from azureml.core import Model
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.image import ContainerImage
from azureml.core.image.container import ContainerImageConfig

from azure_utils.configuration.notebook_config import project_configuration_file
from azure_utils.configuration.project_configuration import ProjectConfiguration
from azure_utils.machine_learning.utils import get_or_create_workspace_from_project
from azure_utils.utilities import text_to_json


def get_or_create_image(configuration_file: str = project_configuration_file, show_output: bool = True,
                        model_name='question_match_model') -> ContainerImage:
    """
    Get or Create new Docker Image from Machine Learning Workspace

    :param configuration_file: path to project configuration file. default: project.yml
    :param show_output: toggle on/off standard output. default: `True`
    :param model_name: Name of Model to package with Image from Machine Learning Workspace
    :return: New or Existing Docker Image for deployment to Kubernetes Compute
    """
    project_configuration = ProjectConfiguration(configuration_file)
    workspace = get_or_create_workspace_from_project(project_configuration, show_output=show_output)

    image_name = project_configuration.get_value("image_name")

    workspace_images = workspace.images
    if image_name in workspace_images:
        return workspace_images[image_name]

    models = [Model(workspace, name=model_name)]

    image_config = create_lightgbm_image_config()

    image_create_start = time.time()

    image = ContainerImage.create(name=image_name, models=models, image_config=image_config, workspace=workspace)
    image.wait_for_creation(show_output=show_output)
    if show_output:
        deployment_time_secs = str(time.time() - image_create_start)
        print("Deployed Image with name " + image_name + ". Took " + deployment_time_secs + " seconds.")
        print(image.name)
        print(image.version)
        print(image.image_build_log_uri)
    return image


def create_lightgbm_image_config(conda_file="lgbmenv.yml", execution_script="score.py") -> ContainerImageConfig:
    """
    Image Configuration for running LightGBM in Azure Machine Learning Workspace

    :param conda_file: file name of LightGBM Conda Env File. This file is created if it does not exist.
     default: lgbmenv.yml
    :param execution_script: webservice file. default: score.py
    :return: new image configuration for Machine Learning Workspace
    """
    create_lightgbm_conda_file(conda_file)

    return ContainerImage.image_configuration(
        execution_script=execution_script,
        runtime="python",
        conda_file=conda_file,
        description="Image with lightgbm model",
        tags={
            "area": "text",
            "type": "lightgbm"
        },
        dependencies=[
            "./notebooks/data_folder/questions.tsv"
        ],
    )


def create_lightgbm_conda_file(conda_file="lgbmenv.yml"):
    """
    Create new Conda File with LightGBM requirements.

    :param conda_file: filename of LightGBM conda file, which is created during call.
    """
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
    with open(conda_file, "w") as file:
        file.write(lgbmenv.serialize_to_string())


def lightgbm_test_image_locally(image, directory):
    """
    Test LightGBM image Locally.

    :param image: Machine Learning Image to test.
    :param directory: root directory that contains data directory.
    """
    dupes_test_path = directory + '/data_folder/dupes_test.tsv'
    dupes_test = pd.read_csv(dupes_test_path, sep='\t', encoding='latin1')
    text_to_score = dupes_test.iloc[0, 4]
    json_text = text_to_json(text_to_score)
    image.run(input_data=json_text)

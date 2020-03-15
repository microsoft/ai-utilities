"""
AI-Utilities - create_deep_model.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import numpy as np
import wget
from PIL import Image, ImageOps
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.image import ContainerImage

from azure_utils.configuration.notebook_config import project_configuration_file
from azure_utils.machine_learning.realtime.image import get_or_create_image


def download_test_image():
    """

    :return:
    """
    wget.download("https://bostondata.blob.core.windows.net/aksdeploymenttutorialaml/220px-Lynx_lynx_poing.jpg")
    img_path = "220px-Lynx_lynx_poing.jpg"
    print(Image.open(img_path).size)
    Image.open(img_path)
    # Below, we load the image by resizing to (224, 224) and then preprocessing using the methods from keras
    # preprocessing and imagenet utilities.
    # Evaluate the model using the input data
    img = Image.open(img_path).convert("RGB")
    img = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)
    img = np.array(img)  # shape: (224, 224, 3)
    img = np.expand_dims(img, axis=0)
    from keras.applications.imagenet_utils import preprocess_input
    return preprocess_input(img)


def get_or_create_resnet_image(configuration_file: str = project_configuration_file, show_output=True,
                               models: list = None, image_settings_name="deep_image_name"):
    """
    Build Image

    :param models:
    :param configuration_file: path to project configuration file. default: project.yml
    :param show_output: toggle on/off standard output. default: `True`
    :param image_settings_name: Setting from Project Configuration
    """
    image_config = create_resnet_image_config()

    return get_or_create_image(image_config, image_settings_name, show_output, models, configuration_file)


def create_resnet_image_config(conda_file="img_env.yml", execution_script="driver.py"):
    """

    :param conda_file:
    :param execution_script:
    :return:
    """
    conda_pack = ["tensorflow-gpu==1.14.0"]
    requirements = ["keras==2.2.0", "Pillow==5.2.0", "azureml-defaults", "azureml-contrib-services", "toolz==0.9.0"]
    imgenv = CondaDependencies.create(conda_packages=conda_pack, pip_packages=requirements)
    with open("img_env.yml", "w") as file:
        file.write(imgenv.serialize_to_string())

    description = "Image for AKS Deployment Tutorial"
    dependencies = ["resnet152.py"]
    tags = {"name": "AKS", "project": "AML"}
    return ContainerImage.image_configuration(execution_script=execution_script, runtime="python",
                                              conda_file=conda_file, description=description, tags=tags,
                                              dependencies=dependencies, enable_gpu=True)

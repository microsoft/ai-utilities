"""
AI-Utilities - create_deep_model.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import numpy as np
import wget
from PIL import Image, ImageOps
from azureml.core import ComputeTarget
from azureml.core.compute import AksCompute
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.image import ContainerImage
from azureml.core.model import Model
from azureml.core.webservice import Webservice, AksWebservice
from keras import backend
from keras.applications.imagenet_utils import preprocess_input
from resnet import ResNet152

from azure_utils.configuration.notebook_config import project_configuration_file
from azure_utils.configuration.project_configuration import ProjectConfiguration
from azure_utils.machine_learning.realtime.image import get_model, has_model, get_or_create_image
from azure_utils.machine_learning.utils import get_or_create_workspace_from_project


def create_deep_model(configuration_file: str = project_configuration_file):
    """

    :param configuration_file:
    :return:
    """
    project_configuration = ProjectConfiguration(configuration_file)

    # If you see error msg "InternalError: Dst tensor is not initialized.", it indicates there are not enough memory.
    model = has_model("resnet_model")
    if model:
        return get_model("resnet_model")
    model = ResNet152(weights="imagenet")
    print("model loaded")

    download_test_image()

    # ## Register the model
    # Register an existing trained model, add descirption and tags.

    # Get workspace
    # Load existing workspace from the config file info.
    workspace = get_or_create_workspace_from_project(project_configuration)
    print(workspace.name, workspace.resource_group, workspace.location, workspace.subscription_id, sep="\n")

    model.save_weights("model_resnet_weights.h5")

    # Register the model

    model = Model.register(
        model_path="model_resnet_weights.h5",  # this points to a local file
        model_name="resnet_model",  # this is the name the   model is registered as
        tags={"model": "dl", "framework": "resnet"},
        description="resnet 152 model",
        workspace=workspace,
    )

    print(model.name, model.description, model.version)

    # Clear GPU memory
    backend.clear_session()
    return model


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
    return preprocess_input(img)


def develop_model_driver():
    """
    Develop Model Driver

    In this notebook, we will develop the API that will call our model. This module initializes the model,
    transforms the input so that it is in the appropriate format and defines the scoring method that will produce
    the predictions. The API will expect the input to be passed as an image. Once a request is received,
    the API will convert load the image preprocess it and pass it to the model. There are two main functions in the
    API: init() and run(). The init() function loads the model and returns a scoring function. The run() function
    processes the images and uses the first function to score them.

        Note: Always make sure you don't have any lingering notebooks running (Shutdown previous notebooks).
        Otherwise it may cause GPU memory issue.

    """
    # ## Write and save driver script
    with open('driver.py', "w") as file:
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


def get_or_create_resnet_image(configuration_file: str = project_configuration_file, show_output=True,
                               image_settings_name="deep_image_name"):
    """
    Build Image

    :param configuration_file: path to project configuration file. default: project.yml
    :param show_output: toggle on/off standard output. default: `True`
    :param image_settings_name: Setting from Project Configuration
    """
    image_config = create_resnet_image_config()

    project_configuration = ProjectConfiguration(configuration_file)
    workspace = get_or_create_workspace_from_project(project_configuration)

    model_name = 'resnet_model'
    models = [workspace.models[model_name]]

    return get_or_create_image(image_config, image_settings_name, models, show_output, configuration_file)


def create_resnet_image_config():
    conda_pack = ["tensorflow-gpu==1.14.0"]
    requirements = ["keras==2.2.0", "Pillow==5.2.0", "azureml-defaults", "azureml-contrib-services", "toolz==0.9.0"]
    imgenv = CondaDependencies.create(conda_packages=conda_pack, pip_packages=requirements)
    with open("img_env.yml", "w") as file:
        file.write(imgenv.serialize_to_string())

    image_config = ContainerImage.image_configuration(execution_script="driver.py", runtime="python",
                                                      conda_file="img_env.yml",
                                                      description="Image for AKS Deployment Tutorial",
                                                      tags={"name": "AKS", "project": "AML"},
                                                      dependencies=["resnet152.py"], enable_gpu=True)
    return image_config


def deploy_on_aks(configuration_file: str = project_configuration_file):
    """

    :param configuration_file:
    :return:
    """
    project_configuration = ProjectConfiguration(configuration_file)
    assert project_configuration.has_settings("deep_image_name")
    assert project_configuration.has_settings("deep_aks_service_name")
    assert project_configuration.has_settings("deep_aks_name")
    assert project_configuration.has_settings("deep_aks_location")

    image_name = project_configuration.get_value("deep_image_name")
    aks_service_name = project_configuration.get_value("deep_aks_service_name")

    ws = get_or_create_workspace_from_project(project_configuration)
    print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep="\n")
    if aks_service_name in ws.webservices:
        return ws.webservices[aks_service_name]

    node_count = 3  # We need to have a minimum of 3 nodes

    aks_target = get_or_create_deep_aks()

    # Deploy web service to AKS
    # Set the web service configuration (using customized configuration)
    aks_config = AksWebservice.deploy_configuration(autoscale_enabled=False, num_replicas=node_count)

    # get the image built in previous notebook
    if image_name not in ws.images:
        get_or_create_resnet_image()
    image = ws.images[image_name]

    aks_service = Webservice.deploy_from_image(
        workspace=ws,
        name=aks_service_name,
        image=image,
        deployment_config=aks_config,
        deployment_target=aks_target,
    )
    aks_service.wait_for_deployment(show_output=True)
    print(aks_service.state)
    return aks_service


def get_or_create_deep_aks(configuration_file: str = project_configuration_file):
    """

    :param configuration_file:
    :return:
    """
    project_configuration = ProjectConfiguration(configuration_file)
    assert project_configuration.has_settings("deep_aks_name")

    aks_name = project_configuration.get_value("deep_aks_name")

    workspace = get_or_create_workspace_from_project(project_configuration)
    workspace_compute = workspace.compute_targets
    if aks_name in workspace_compute:
        return workspace_compute[aks_name]

    prov_config = AksCompute.provisioning_configuration(vm_size="Standard_NC6")
    aks_target = ComputeTarget.create(workspace=workspace, name=aks_name, provisioning_configuration=prov_config)

    aks_target.wait_for_completion(show_output=True)

    return aks_target

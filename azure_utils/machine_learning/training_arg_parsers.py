"""
AI-Utilities - training_arg_parsers.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import argparse
import os
from argparse import Namespace

import numpy as np
from PIL import Image, ImageOps
from azureml.contrib.services.aml_response import AMLResponse
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.preprocessing import image
from toolz import compose


def get_training_parser() -> Namespace:
    """
    Argument Parser for Training Model Scripts

    :return: parsed args
    """
    parser = argparse.ArgumentParser(
        description="Fit and evaluate a model based on train-test datasets."
    )
    parser.add_argument("--outputs", help="the outputs directory", default="outputs")
    parser.add_argument("--model", help="the model file", default="model.pkl")
    return parser.parse_args()


NUMBER_RESULTS = 3


def get_model_path(model_pkl: str = "model.pkl"):
    """
    Get Model Path either locally or in web service

    :param model_pkl: filename of file
    :return: Model Directory
    """
    model_dir = "outputs"
    if os.getenv("AZUREML_MODEL_DIR"):
        model_dir = os.getenv("AZUREML_MODEL_DIR")
    assert os.path.isfile(model_dir + "/" + model_pkl), """Model not found."""
    return model_dir + "/" + model_pkl


def image_ref_to_pil_image(image_ref: str):
    """ Load image with PIL (RGB) """
    return Image.open(image_ref).convert("RGB")


def pil_to_numpy(pil_image):
    """

    :param pil_image:
    :return:
    """
    img = ImageOps.fit(pil_image, (224, 224), Image.ANTIALIAS)
    img = image.img_to_array(img)
    return img


def default_response(request) -> AMLResponse:
    """

    :param request:
    :return:
    """
    if request.method == "GET":
        return AMLResponse({"azEnvironment": "Azure"}, 201)
    return AMLResponse("bad request", 500)


def prepare_response(preds, transformed_dict):
    """

    :param preds:
    :param transformed_dict:
    :return:
    """
    preds = decode_predictions(preds.astype(np.float64), top=NUMBER_RESULTS)
    return dict(zip(transformed_dict.keys(), preds))


def process_request(request):
    """

    :param request:
    :return:
    """
    transform_input = compose(pil_to_numpy, image_ref_to_pil_image)
    transformed_dict = {
        key: transform_input(img_ref) for key, img_ref in request.files.items()
    }
    img_array = preprocess_input(np.stack(list(transformed_dict.values())))
    return img_array, transformed_dict

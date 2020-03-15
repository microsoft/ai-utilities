# -*- coding: utf-8 -*-
"""ResNet152 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adaptation of code from flyyufelix, mvoelk, BigMoyan, fchollet at https://github.com/adamcasson/resnet152

"""
from typing import Any

from azureml.contrib.services.aml_response import AMLResponse
from azureml.core import Model


class RTSEstimator:
    """Estimator for Real-time Scoring"""

    def predict(self, request) -> AMLResponse:
        """

        :param request:
        """
        raise NotImplementedError

    def load_model(self):
        """
        Abstract Method for load model
        """
        raise NotImplementedError

    def save_model(self, path: str):
        """
        Abstract Method for Save Model
        """
        raise NotImplementedError

    def train(self):
        """
        Abstract Method for Train Model
        """
        raise NotImplementedError

    def create_model(
        self,
        include_top: bool = True,
        weights: str = None,
        input_tensor: Any = None,
        input_shape: Any = None,
        large_input: bool = False,
        pooling: Any = None,
        classes: int = 1000,
        save_model: bool = False,
        model_path: str = None,
    ) -> Model:
        """
        Abstract Method for Create Model

        :param include_top:
        :param weights:
        :param input_tensor:
        :param input_shape:
        :param large_input:
        :param pooling:
        :param classes:
        :param save_model:
        :param model_path:
        """
        raise NotImplementedError

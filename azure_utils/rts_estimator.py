# -*- coding: utf-8 -*-
"""ResNet152 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adaptation of code from flyyufelix, mvoelk, BigMoyan, fchollet at https://github.com/adamcasson/resnet152

"""
from azureml.contrib.services.aml_response import AMLResponse


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

    def create_model(self, **kwargs):
        """
        Abstract Method for Create Model

        :param kwargs: dict of arguments
        """
        raise NotImplementedError

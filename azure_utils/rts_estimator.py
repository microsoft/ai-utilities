# -*- coding: utf-8 -*-
"""ResNet152 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adaptation of code from flyyufelix, mvoelk, BigMoyan, fchollet at https://github.com/adamcasson/resnet152

"""


class RTSEstimator:
    def predict(self, request):
        """

        :param request:
        """
        raise NotImplementedError

    def load_model(self):
        """

        """
        raise NotImplementedError

    def save_model(self, path):
        """

        """
        raise NotImplementedError

    def train(self):
        """

        """
        raise NotImplementedError

    def create_model(self, **kwargs):
        """

        :param kwargs:
        """
        raise NotImplementedError

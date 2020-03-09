"""
AI-Utilities - realtime_factory.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import inspect

from azureml.contrib.services import rawhttp


class RealTimeFactory:
    """

    Example Usage:
        from azure_utils.machine_learning.factories.realtime_factory import RealTimeFactory

        rts_factory = RealTimeFactory()
        init = rts_factory.score_init
        run = rts_factory.score_run
    or
        from azure_utils.machine_learning.factories.realtime_factory import RealTimeFactory
        from azure_utils.machine_learning.models.training_arg_parsers import get_training_parser

        if __name__ == '__main__':
            RealTimeFactory().train(get_training_parser())

    """

    def __init__(self):
        raise NotImplementedError

    def train(self, args):
        """
        Train Abstract Method
        :param args:
        """
        raise NotImplementedError

    def score_init(self):
        """
        Score Init Abstract Method
        """
        raise NotImplementedError

    @rawhttp
    def score_run(self, request):
        """
        Score Run Abstract Method

        :param request:
        """
        raise NotImplementedError

    @classmethod
    def make_file(cls):
        """
        Make file from class

        :return: string of file of class
        """
        file = inspect.getsource(cls)

        file = file.replace(
            inspect.getsource(RealTimeFactory.train), inspect.getsource(cls.train)
        )
        file = file.replace(
            inspect.getsource(RealTimeFactory.score_init),
            inspect.getsource(cls.score_init),
        )
        file = file.replace(
            inspect.getsource(RealTimeFactory.score_run),
            inspect.getsource(cls.score_run),
        )
        file = file.replace(
            inspect.getsource(RealTimeFactory.__init__), inspect.getsource(cls.__init__)
        )
        file = file.replace("RealTimeFactory", "DeepRealTimeFactory(RealTimeFactory)")
        return file

"""
AI-Utilities - realtime_factory.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import inspect


class RealTimeFactory:
    from azureml.contrib.services import rawhttp
    def __init__(self):
        raise NotImplementedError

    def train(self, args):
        raise NotImplementedError

    def score_init(self):
        raise NotImplementedError

    @rawhttp
    def score_run(self, request):
        raise NotImplementedError

    @classmethod
    def make_file(cls):
        file = inspect.getsource(cls)

        file = file.replace(inspect.getsource(RealTimeFactory.train), inspect.getsource(cls.train))
        file = file.replace(inspect.getsource(RealTimeFactory.score_init),
                            inspect.getsource(cls.score_init))
        file = file.replace(inspect.getsource(RealTimeFactory.score_run), inspect.getsource(cls.score_run))
        file = file.replace(inspect.getsource(RealTimeFactory.__init__), inspect.getsource(cls.__init__))
        file = file.replace("RealTimeFactory", "DeepRealTimeFactory(RealTimeFactory)")
        file

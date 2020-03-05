from azure_utils.machine_learning.factories.realtime_factory import RealTimeFactory
from azure_utils.machine_learning.models.training_arg_parsers import get_training_parser

if __name__ == '__main__':
    RealTimeFactory().train(get_training_parser())

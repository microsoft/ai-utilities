"""
AI-Utilities - create_deep_model.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import os

from azure_utils.machine_learning.training_arg_parsers import get_training_parser
from azure_utils.samples.deep_rts_samples import ResNet152

if __name__ == '__main__':
    args = get_training_parser()

    ResNet152().create_model(weights="imagenet", save_model=True, model_path=os.path.join(args.outputs, args.model))

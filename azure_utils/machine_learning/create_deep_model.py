"""
ai-utilities - machine_learning/create_model.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import argparse
import os

from azureml.core import Run
from sklearn.externals import joblib

from resnet152 import ResNet152

if __name__ == '__main__':
    # """ Main Method to use with AzureML"""
    # Define the arguments.
    parser = argparse.ArgumentParser(description='Fit and evaluate a model based on train-test datasets.')
    parser.add_argument('-v', '--verbose', help='the verbosity of the estimator', type=int, default=-1)
    parser.add_argument('--outputs', help='the outputs directory', default='.')
    parser.add_argument('-s', '--save', help='save the model', action='store_true', default=True)
    parser.add_argument('--model', help='the model file', default='model.pkl')
    args = parser.parse_args()

    run = Run.get_context()
    outputs_path = args.outputs
    os.makedirs(outputs_path, exist_ok=True)
    model_path = os.path.join(outputs_path, args.model)

    resnet_152 = ResNet152(weights="imagenet")

    # Save the model to a file, and report on its size.
    if args.save:
        joblib.dump(resnet_152, resnet_152.get_weights())

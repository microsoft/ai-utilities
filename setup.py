"""
AI-Utilities - setup.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from setuptools import find_packages, setup

# This is the name of your PyPI-package.
setup(name="Microsoft-AI-Azure-Utility-Samples", version="0.3.81", description="Utility Samples for AI Solutions",
      author="Daniel Ciborowski & Daniel Grecoe", author_email="dciborow@microsoft.com",
      url="https://github.com/microsoft/AI-Utilities", license="MIT", packages=find_packages(),
      install_requires=['azureml-core', 'azure-cli==2.0.81', 'python-dotenv', 'nbformat', 'papermill', 'nbconvert',
                        'junit_xml', 'PyYAML', 'pytest', 'lightgbm==2.1.2', 'pandas', 'sklearn', 'msrestazure',
                        'azure-common', 'azure-cli-ml', 'deprecated', 'requests', 'wget', 'numpy', 'Pillow', 'keras',
                        'resnet', 'azureml-contrib-functions', 'toolz', 'knack', 'ipywidgets', 'tabulate', 'flask'])

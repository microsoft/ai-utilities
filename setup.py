"""
AI-Utilities - setup.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from setuptools import find_packages, setup

# This is the name of your PyPI-package.
setup(name="Microsoft-AI-Azure-Utility-Samples", version="0.3.9", description="Utility Samples for AI Solutions",
      author="Daniel Ciborowski & Daniel Grecoe", author_email="dciborow@microsoft.com",
      url="https://github.com/microsoft/AI-Utilities", license="MIT", packages=find_packages(),
      install_requires=['lightgbm'])


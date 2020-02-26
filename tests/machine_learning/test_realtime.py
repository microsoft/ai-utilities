"""
AI-Utilities - test_realtime.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

This are long tests and are not currently tested in this SDK.
"""
from azure_utils.machine_learning.realtime.image import get_or_create_image
from azure_utils.machine_learning.realtime.kubernetes import get_or_create_aks_service


def get_or_create_image_test():
    """Test Get or Create Machine Learning Docker Image"""
    image = get_or_create_image()
    assert image


def get_or_create_aks_service_test():
    """Test Get Or Create Kubernetes Compute and Web Service"""
    aks_webservice = get_or_create_aks_service()
    assert aks_webservice

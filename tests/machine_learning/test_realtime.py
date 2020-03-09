"""
AI-Utilities - test_realtime.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

This are long tests and are not currently tested in this SDK.
"""
from azure_utils.machine_learning.realtime.image import get_or_create_lightgbm_image
from azure_utils.machine_learning.utils import get_or_create_workspace_from_file


def dont_test_get_or_create_workspace():
    """Test Get or Create Machine Learning Workspace"""
    get_or_create_workspace_from_file()


def dont_test_get_or_create_image():
    """Test Get or Create Machine Learning Docker Image"""
    image = get_or_create_lightgbm_image()
    assert image.creation_state != "Failed"

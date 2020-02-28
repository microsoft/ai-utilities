"""
AI-Utilities - test_realtime.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

This are long tests and are not currently tested in this SDK.
"""
from azure_utils.machine_learning.realtime.image import get_or_create_image
from azure_utils.machine_learning.realtime.kubernetes import get_or_create_aks_service, get_or_create_aks
from azure_utils.machine_learning.utils import get_or_create_workspace_from_file


def test_get_or_create_workspace():
    """Test Get or Create Machine Learning Workspace"""
    get_or_create_workspace_from_file()


def test_get_or_create_image():
    """Test Get or Create Machine Learning Docker Image"""
    image = get_or_create_image()
    assert image.creation_state != "Failed"


def test_get_or_create_aks_service():
    """Test Get Or Create Kubernetes Compute and Web Service"""

    score_py = """
import json
import logging


def init():
    logger = logging.getLogger("scoring_script")
    logger.info("init")


def run(body):
    logger = logging.getLogger("scoring_script")
    logger.info("run")
    return json.dumps({'call': True})

"""
    with open("score.py", "w") as file:
        file.write(score_py)
    get_or_create_image()
    aks_webservice = get_or_create_aks_service()
    assert aks_webservice


def test_get_or_create_aks():
    """ Test Get or Create AKS"""
    get_or_create_image()
    get_or_create_aks()

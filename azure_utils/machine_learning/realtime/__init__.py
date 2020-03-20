"""
AI-Utilities - __init__.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from azureml.contrib.services.aml_response import AMLResponse


def default_response(request) -> AMLResponse:
    """

    :param request:
    :return:
    """
    if request.method == "GET":
        return AMLResponse({"azEnvironment": "Azure"}, 201)
    return AMLResponse("bad request", 500)

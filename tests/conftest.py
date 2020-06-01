"""
AI-Utilities - tests/conftest.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import pytest


def pytest_addoption(parser):
    parser.addoption("--iterations", action="store", default=5)
    parser.addoption("--threads", action="store", default=2)


@pytest.fixture
def threads(request):
    return int(request.config.getoption("--threads"))


@pytest.fixture
def iterations(request):
    return int(request.config.getoption("--iterations"))


def pytest_collection_modifyitems(items):
    for item in items:
        if "mock" in item.nodeid:
            item.add_marker(pytest.mark.interface)
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.interface)

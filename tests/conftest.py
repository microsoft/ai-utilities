"""
AI-Utilities - tests/conftest.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "mock" in item.nodeid:
            item.add_marker(pytest.mark.interface)
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.interface)

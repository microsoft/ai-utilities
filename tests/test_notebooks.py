"""
AI-Utilities - test_notebooks.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import pytest

from azure_utils import notebook_directory
from azure_utils.dev_ops.testing_utilities import run_notebook



@pytest.mark.parametrize("notebook", [notebook_directory + "/exampleconfiguration.ipynb"])
def dont_test_notebook(notebook: str, add_nunit_attachment: pytest.fixture):
    """
    Jupyter Notebook Test
    :param notebook: input notebook
    :param add_nunit_attachment: pytest fixture
    """
    run_notebook(notebook, add_nunit_attachment, kernel_name="ai-utilities", root=notebook_directory)

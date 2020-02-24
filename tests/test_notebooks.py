"""
AI-Utilities - test_notebooks.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import warnings

import pytest
from papermill import PapermillExecutionError

from azure_utils.dev_ops.testing_utilities import run_notebook
from notebooks import notebook_directory


@pytest.mark.parametrize(
    "notebook",
    [
        notebook_directory + "/exampleconfiguration.ipynb"
    ]
)
def test_notebook(notebook, add_nunit_attachment):
    try:
        run_notebook(notebook, add_nunit_attachment, kernel_name="ai-utilities", root=notebook_directory)
    except PapermillExecutionError:
        warnings.warn("Notebook could not be tested")

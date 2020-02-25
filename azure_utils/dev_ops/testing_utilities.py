"""
AI-Utilities - testing_utilities.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import json
import os
import re
import sys

import nbformat
from nbconvert import MarkdownExporter, RSTExporter
from junit_xml import TestCase, TestSuite
import papermill as pm

notebook_output_ext = ".output_ipynb"


def run_notebook(input_notebook, add_nunit_attachment, parameters=None, kernel_name="ai-architecture-template",
                 root="."):
    """
    Used to run a notebook in the correct directory.

    Parameters
    ----------
    :param input_notebook: Name of Notebook to Test
    :param add_nunit_attachment:
    :param parameters:
    :param kernel_name: Jupyter Kernal
    :param root:
    """

    output_notebook = input_notebook.replace(".ipynb", notebook_output_ext)
    try:
        results = pm.execute_notebook(
            os.path.join(root, input_notebook),
            os.path.join(root, output_notebook),
            parameters=parameters,
            kernel_name=kernel_name
        )

        for cell in results.cells:
            if cell.cell_type == "code":
                assert not cell.metadata.papermill.exception, "Error in Python Notebook"
    finally:
        with open(os.path.join(root, output_notebook)) as json_file:
            data = json.load(json_file)
            jupyter_output = nbformat.reads(json.dumps(data), as_version=nbformat.NO_CONVERT)

        export_md(jupyter_output, output_notebook, add_nunit_attachment, file_ext=".txt", root=root)

        regex = r'Deployed (.*) with name (.*). Took (.*) seconds.'

        with open(os.path.join(root, output_notebook), 'r') as file:
            data = file.read()

            test_cases = []
            for group in re.findall(regex, data):
                test_cases.append(
                    TestCase(name=group[0] + " creation", classname=input_notebook, elapsed_sec=float(group[2]),
                             status="Success"))

            test_suite = TestSuite("my test suite", test_cases)

            with open('test-timing-output.xml', 'w') as file:
                TestSuite.to_file(file, [test_suite], prettyprint=False)


def export_notebook(exporter, jupyter_output, output_notebook, add_nunit_attachment, file_ext, root='.'):
    """
    Export Jupyter Output to File

    :param exporter:
    :param jupyter_output:
    :param output_notebook:
    :param add_nunit_attachment:
    :param file_ext:
    :param root:
    """
    (body, _) = exporter.from_notebook_node(jupyter_output)
    with open(os.path.join(root, output_notebook.replace(notebook_output_ext, file_ext)), "w") as text_file:
        sys.stderr.write(body)
        text_file.write(body)

    if add_nunit_attachment is not None:
        path = os.path.join(root, output_notebook.replace(notebook_output_ext, file_ext))
        add_nunit_attachment(path, output_notebook)


def export_md(jupyter_output, output_notebook, add_nunit_attachment, file_ext='.md', root="."):
    """
    Export Jupyter Output to Markdown File

    :param jupyter_output:
    :param output_notebook:
    :param add_nunit_attachment:
    :param file_ext:
    :param root:
    """
    markdown_exporter = MarkdownExporter()
    export_notebook(markdown_exporter, jupyter_output, output_notebook, add_nunit_attachment, file_ext, root=root)


def export_rst(jupyter_output, output_notebook, add_nunit_attachment, file_ext='.rst', root="."):
    """
    Export Jupyter Output to RST File

    :param jupyter_output:
    :param output_notebook:
    :param add_nunit_attachment:
    :param file_ext:
    :param root:
    """
    rst_exporter = RSTExporter()
    export_notebook(rst_exporter, jupyter_output, output_notebook, add_nunit_attachment, file_ext, root=root)

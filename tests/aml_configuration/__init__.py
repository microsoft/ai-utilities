"""
ado-ml-batch-train - __init__.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import os

if os.path.exists(os.path.join(os.curdir, "aml_configuration")):
    os.chdir("aml_configuration")

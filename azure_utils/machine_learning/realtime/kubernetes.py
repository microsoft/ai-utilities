"""
AI-Utilities - kubernetes.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from azureml.core import ComputeTarget
from azureml.core.compute import AksCompute


def create_aks():
    prov_config = AksCompute.provisioning_configuration(agent_count=node_count,
                                                        vm_size=vm_size,
                                                        location=aks_location)

    # Create the cluster
    aks_target = ComputeTarget.create(workspace=ws,
                                      name=aks_name,
                                      provisioning_configuration=prov_config)

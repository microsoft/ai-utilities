"""
AI-Utilities - kubernetes.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import time

from azureml.core import ComputeTarget, Webservice
from azureml.core.compute import AksCompute
from azureml.core.webservice import AksWebservice

from azure_utils.configuration.notebook_config import project_configuration_file
from azure_utils.configuration.project_configuration import ProjectConfiguration
from azure_utils.machine_learning.utils import get_or_create_workspace_from_project


def create_deploy_aks_service(configuration_file: str = "../" + project_configuration_file,
                              vm_size: str = "Standard_D4_v2", node_count: int = 4, num_replicas: int = 2,
                              cpu_cores: int = 1, show_output: bool = True):
    project_configuration = ProjectConfiguration(configuration_file)

    aks_target = get_or_create_aks(project_configuration, show_output, node_count, vm_size)

    return get_or_create_service(project_configuration, aks_target, cpu_cores, show_output, num_replicas)


def get_or_create_service(project_configuration, aks_target, cpu_cores, show_output, num_replicas):
    workspace = get_or_create_workspace_from_project(project_configuration, show_output=show_output)

    aks_service_name = project_configuration.get_value("aks_service_name")
    image_name = project_configuration.get_value("image_name")

    web_service = Webservice._get(aks_service_name)
    if web_service:
        return web_service

    aks_config = AksWebservice.deploy_configuration(num_replicas=num_replicas, cpu_cores=cpu_cores)
    image = workspace.images[image_name]
    deploy_from_image_start = time.time()
    aks_service = Webservice.deploy_from_image(workspace=workspace, name=aks_service_name, image=image,
                                               deployment_config=aks_config, deployment_target=aks_target)
    aks_service.wait_for_deployment(show_output=show_output)
    if show_output:
        deployment_time_secs = str(time.time() - deploy_from_image_start)
        print("Deployed Image with name "
              + aks_service_name + ". Took " + deployment_time_secs + " seconds.")
        print(aks_service.state)
        print(aks_service.get_logs())
    return aks_service


def get_or_create_aks(project_configuration, show_output, node_count, vm_size):
    workspace = get_or_create_workspace_from_project(project_configuration, show_output=show_output)

    aks_name = project_configuration.get_value("aks_name")
    aks_service_name = project_configuration.get_value("aks_service_name")
    aks_location = project_configuration.get_value("workspace_region")

    workspace_compute = workspace.compute_targets
    if aks_name in workspace_compute:
        return workspace_compute[aks_name]

    prov_config = AksCompute.provisioning_configuration(agent_count=node_count, vm_size=vm_size, location=aks_location)

    deploy_aks_start = time.time()
    aks_target: AksCompute = ComputeTarget.create(workspace=workspace, name=aks_name,
                                                  provisioning_configuration=prov_config)
    aks_target.wait_for_completion(show_output=show_output)
    if show_output:
        deployment_time_secs = str(time.time() - deploy_aks_start)
        print("Deployed AKS with name "
              + aks_service_name + ". Took " + deployment_time_secs + " seconds.")
        print(aks_target.provisioning_state)
        print(aks_target.provisioning_errors)
    aks_status = aks_target.get_status()
    assert aks_status == 'Succeeded', 'AKS failed to create'
    return aks_target

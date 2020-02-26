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


def get_or_create_aks_service(configuration_file: str = project_configuration_file,
                              vm_size: str = "Standard_D4_v2", node_count: int = 4, num_replicas: int = 2,
                              cpu_cores: int = 1, show_output: bool = True) -> AksWebservice:
    """
    Get or Create AKS Service with new or existing Kubernetes Compute

    :param configuration_file: path to project configuration file. default: project.yml
    :param vm_size: skew of vms in Kubernetes cluster. default: Standard_D4_v2
    :param node_count: number of nodes in Kubernetes cluster. default: 4
    :param num_replicas: number of replicas in Kubernetes cluster. default: 2
    :param cpu_cores: cpu cores for web service. default: 1
    :param show_output: toggle on/off standard output. default: `True`
    :return: New or Existing Kubernetes Web Service
    """
    project_configuration = ProjectConfiguration(configuration_file)

    aks_target = get_or_create_aks(project_configuration, vm_size, node_count, show_output)

    return get_or_create_service(project_configuration, aks_target, num_replicas, cpu_cores, show_output)


def get_or_create_service(project_configuration, aks_target, num_replicas: int = 2,
                          cpu_cores: int = 1, show_output: bool = True) -> AksWebservice:
    """
    Get or Create new Machine Learning Webservice

    :param project_configuration: Project configuration settings container
    :param aks_target: Kubernetes compute to target for deployment
    :param num_replicas: number of replicas in Kubernetes cluster. default: 2
    :param cpu_cores: cpu cores for web service. default: 1
    :param show_output: toggle on/off standard output. default: `True`
    :return: New or Existing Kubernetes Web Service
    """
    workspace = get_or_create_workspace_from_project(project_configuration, show_output=show_output)

    aks_service_name = project_configuration.get_value("aks_service_name")
    image_name = project_configuration.get_value("image_name")

    if aks_service_name in workspace.webservices:
        return workspace.webservices[aks_service_name]

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


def get_or_create_aks(project_configuration, vm_size: str = "Standard_D4_v2", node_count: int = 4,
                      show_output: bool = True) -> AksCompute:
    """
    Get or Create Azure Machine Learning Kubernetes Compute

    :param project_configuration: Project configuration settings container
    :param vm_size: skew of vms in Kubernetes cluster. default: Standard_D4_v2
    :param node_count: number of nodes in Kubernetes cluster. default: 4
    :param show_output: toggle on/off standard output. default: `True`
    :return: New or Existing Kubernetes Compute
    """
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

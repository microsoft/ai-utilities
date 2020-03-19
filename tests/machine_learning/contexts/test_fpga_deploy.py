import wget

from azure_utils.machine_learning.contexts.realtime_score_context import (
    FPGARealtimeScore,
)

from azure_utils.machine_learning.contexts.workspace_contexts import WorkspaceContext
from azureml.core.webservice import AksWebservice
import requests
import pytest


@pytest.fixture
def workspace():
    return WorkspaceContext.get_or_create_workspace()


def test_fpga_deploy(workspace):
    model_name = "resnet50"
    image_name = "{}-image".format(model_name)
    aks_name = "my-aks-cluster"

    image = FPGARealtimeScore.register_resnet_50(workspace, model_name, image_name)
    assert image
    # Create the cluster
    aks_target = FPGARealtimeScore.create_aks(workspace, aks_name)
    assert aks_target
    aks_service = FPGARealtimeScore.create_aks_service(workspace, aks_target, image)
    assert aks_service


def test_gpu_service(workspace):
    aks_service_name = "deepaksservice"

    assert aks_service_name in workspace.webservices, f"{aks_service_name} not found."
    aks_service = AksWebservice(workspace, name=aks_service_name)
    assert aks_service.state == "Succeeded", f"{aks_service_name} is in state {aks_service.state}."
    scoring_url = aks_service.scoring_uri
    print(scoring_url)
    api_key = aks_service.get_keys()[0]
    import requests

    IMAGEURL = "https://bostondata.blob.core.windows.net/aksdeploymenttutorialaml/220px-Lynx_lynx_poing.jpg"
    headers = {"Authorization": ("Bearer " + api_key)}
    import urllib
    import toolz
    from io import BytesIO

    img_data = toolz.pipe(
        IMAGEURL, urllib.request.urlopen, lambda x: x.read(), BytesIO
    ).read()
    r = requests.post(scoring_url, files={"image": img_data}, headers=headers)
    r.json()


def test_fpga_service(workspace):
    # Using the grpc client in Azure ML Accelerated Models SDK package
    aks_service_name = "my-aks-service"
    aks_service = AksWebservice(workspace=workspace, name=aks_service_name)
    client = FPGARealtimeScore.get_prediction_client(aks_service)

    # Score image with input and output tensor names
    input_tensors, output_tensors = FPGARealtimeScore.get_resnet50_IO()
    wget.download(
        "https://raw.githubusercontent.com/Azure/MachineLearningNotebooks/"
        "master/how-to-use-azureml/deployment/accelerated-models/snowleopardgaze.jpg"
    )

    results = client.score_file(
        path="snowleopardgaze.jpg", input_name=input_tensors, outputs=output_tensors
    )

    # map results [class_id] => [confidence]
    results = enumerate(results)
    # sort results by confidence
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    # print top 5 results
    classes_entries = requests.get(
        "https://raw.githubusercontent.com/Lasagne/Recipes/"
        "master/examples/resnet50/imagenet_classes.txt"
    ).text.splitlines()
    for top in sorted_results[:5]:
        print(classes_entries[top[0]], "confidence:", top[1])

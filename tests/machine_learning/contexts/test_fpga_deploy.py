import wget

from azure_utils.machine_learning.contexts.realtime_score_context import (
    FPGARealtimeScore,
)

from azure_utils.machine_learning.contexts.workspace_contexts import WorkspaceContext
from azureml.core.webservice import AksWebservice
import requests


def test_fpga_deploy():
    ws = WorkspaceContext.get_or_create_workspace()
    model_name = "resnet50"
    image_name = "{}-image".format(model_name)
    aks_name = "my-aks-cluster"

    image = FPGARealtimeScore.register_resnet_50(ws, model_name, image_name)
    assert image
    # Create the cluster
    aks_target = FPGARealtimeScore.create_aks(ws, aks_name)
    assert aks_target
    aks_service = FPGARealtimeScore.create_aks_service(ws, aks_target, image)
    assert aks_service


def test_fpga_service():
    # Using the grpc client in Azure ML Accelerated Models SDK package
    ws = WorkspaceContext.get_or_create_workspace()
    aks_service_name = "my-aks-service"

    aks_service = AksWebservice(workspace=ws, name=aks_service_name)
    client = FPGARealtimeScore.get_prediction_client(aks_service)

    classes_entries = requests.get(
        "https://raw.githubusercontent.com/Lasagne/Recipes/"
        "master/examples/resnet50/imagenet_classes.txt"
    ).text.splitlines()

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
    for top in sorted_results[:5]:
        print(classes_entries[top[0]], "confidence:", top[1])

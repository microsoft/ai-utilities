import os.path

import pytest
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
from azureml.core.authentication import InteractiveLoginAuthentication
from msrest.authentication import CognitiveServicesCredentials


SUBSCRIPTION_KEY_ENV_NAME = "0ca618d2-22a8-413a-96d0-0f1b531129c3"
COMPUTERVISION_LOCATION = os.environ.get("COMPUTERVISION_LOCATION", "eastus")

IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images")


@pytest.fixture
def subscription_key():
    return SUBSCRIPTION_KEY_ENV_NAME


@pytest.fixture
def computer_vision_client(subscription_key):
    cog_services_mgmt = CognitiveServicesManagementClient(
        InteractiveLoginAuthentication(), subscription_key, base_url=None
    )
    rg_name = "aiutilities-eastus-stable"
    name = "dcib-utilities-east"
    sku = "S0"
    location = "eastus"
    kind = "cognitiveservices"

    cog_services_mgmt.accounts.create(
        rg_name,
        name,
        account={
            "sku": {"name": sku},
            "location": location,
            "kind": kind,
            "properties": {},
        },
    )
    return ComputerVisionClient(
        endpoint="https://eastus.api.cognitive.microsoft.com/",
        credentials=CognitiveServicesCredentials(
            cog_services_mgmt.accounts.list_keys(rg_name, name).key1
        ),
    )


def test_create_cog_service(subscription_key):
    cog_services_mgmt = CognitiveServicesManagementClient(
        InteractiveLoginAuthentication(), subscription_key, base_url=None
    )
    rg_name = "aiutilities-eastus-stable"
    name = "dcib-utilities-east"
    assert cog_services_mgmt.accounts.list_keys(
        rg_name, name
    ).key1, "Key not retrievable"


def test_image_analysis_in_stream(computer_vision_client):
    """ImageAnalysisInStream.

    This will analyze an image from a stream and return all available features.
    """
    with open(os.path.join(IMAGES_FOLDER, "house.jpg"), "rb") as image_stream:
        image_analysis = computer_vision_client.analyze_image_in_stream(
            image=image_stream,
            visual_features=[
                VisualFeatureTypes.image_type,  # Could use simple str "ImageType"
                VisualFeatureTypes.faces,  # Could use simple str "Faces"
                VisualFeatureTypes.categories,  # Could use simple str "Categories"
                VisualFeatureTypes.color,  # Could use simple str "Color"
                VisualFeatureTypes.tags,  # Could use simple str "Tags"
                VisualFeatureTypes.description,  # Could use simple str "Description"
            ],
        )

    print(
        "This image can be described as: {}\n".format(
            image_analysis.description.captions[0].text
        )
    )

    print("Tags associated with this image:\nTag\t\tConfidence")
    for tag in image_analysis.tags:
        print("{}\t\t{}".format(tag.name, tag.confidence))

    print(
        "\nThe primary colors of this image are: {}".format(
            image_analysis.color.dominant_colors
        )
    )


def test_recognize_text(computer_vision_client):
    """RecognizeTextUsingRecognizeAPI.

    This will recognize text of the given image using the recognizeText API.
    """
    import time

    with open(
        os.path.join(IMAGES_FOLDER, "make_things_happen.jpg"), "rb"
    ) as image_stream:
        job = computer_vision_client.recognize_text_in_stream(
            image=image_stream, mode="Printed", raw=True
        )
    operation_id = job.headers["Operation-Location"].split("/")[-1]

    image_analysis = computer_vision_client.get_text_operation_result(operation_id)
    while image_analysis.status in ["NotStarted", "Running"]:
        time.sleep(1)
        image_analysis = computer_vision_client.get_text_operation_result(operation_id=operation_id)

    print("Job completion is: {}\n".format(image_analysis.status))

    print("Recognized:\n")
    lines = image_analysis.recognition_result.lines
    print(lines[0].words[0].text)  # "make"
    print(lines[1].words[0].text)  # "things"
    print(lines[2].words[0].text)  # "happen"


def test_recognize_printed_text_in_stream(computer_vision_client):
    """RecognizedPrintedTextUsingOCR_API.

    This will do an OCR analysis of the given image.
    """

    with open(
        os.path.join(IMAGES_FOLDER, "computer_vision_ocr.png"), "rb"
    ) as image_stream:
        image_analysis = computer_vision_client.recognize_printed_text_in_stream(
            image=image_stream, language="en"
        )

    lines = image_analysis.regions[0].lines
    print("Recognized:\n")
    for line in lines:
        line_text = " ".join([word.text for word in line.words])
        print(line_text)

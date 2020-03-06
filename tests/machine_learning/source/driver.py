
from azure_utils.resnet152 import ResNet152
from azure_utils.machine_learning.models.training_arg_parsers import default_response
from azureml.contrib.services.aml_request import rawhttp


def init():
    """ Initialise the model and scoring function"""
    global resnet_152
    resnet_152 = ResNet152.load_model()

@rawhttp
def run(request):
    """ Make a prediction based on the data passed in using the preloaded model"""
    if request.method == 'POST':
        """ Classify the input using the loaded model"""
        return resnet_152.predict(request)
    return default_response(request)

                    

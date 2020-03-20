
import sys
sys.setrecursionlimit(3000)

from azureml.contrib.services.aml_request import rawhttp

def init():
    """ Initialise the model and scoring function
    """
    global process_and_score
    from azure_utils.samples.deep_rts_samples import get_model_api
    process_and_score = get_model_api()


@rawhttp
def run(request):
    """ Make a prediction based on the data passed in using the preloaded model
    """
    from azure_utils.machine_learning.realtime import default_response
    if request.method == 'POST':
        return process_and_score(request.files)
    return default_response(request)
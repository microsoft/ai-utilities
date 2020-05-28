from azure_utils.samples.deep_rts_samples import _image_ref_to_pil_image, _pil_to_numpy
from toolz import compose
import wget

def test_image_ref_to_pil_image():
    IMAGEURL = "https://bostondata.blob.core.windows.net/aksdeploymenttutorialaml/220px-Lynx_lynx_poing.jpg"

    import urllib
    import toolz
    from io import BytesIO
    img_data = toolz.pipe(IMAGEURL, urllib.request.urlopen, lambda x: x.read(), BytesIO).read()
    transform_input = compose(_pil_to_numpy, _image_ref_to_pil_image)
    # transform_input(img_data)
    wget.download("https://raw.githubusercontent.com/Azure/MachineLearningNotebooks/master/how-to-use-azureml/deployment/accelerated-models/snowleopardgaze.jpg", "snowleopardgaze.jpg")
    images_dict = {"lynx": open("snowleopardgaze.jpg", "rb")}
    transformed_dict = {key: transform_input(img_ref) for key, img_ref in images_dict.items()}
    # _pil_to_numpy(img_data)
    # _image_ref_to_pil_image(_pil_to_numpy(img_data))

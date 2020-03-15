"""
AI-Utilities - kubernetes.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import pandas as pd
import requests
from azureml.core.webservice import AksWebservice

from azure_utils.utilities import text_to_json


def test_aks(directory: str, aks_service: AksWebservice):
    """
    Test AKS with sample call.

    :param directory: directory of data_folder with test data
    :param aks_service: AKS Web Service to Test
    """
    num_dupes_to_score = 4

    dupes_test = get_dupes_test(directory)
    text_to_score = dupes_test.iloc[0, num_dupes_to_score]

    json_text = text_to_json(text_to_score)

    scoring_url = aks_service.scoring_uri
    api_key = aks_service.get_keys()[0]

    headers = {
        "content-type": "application/json",
        "Authorization": ("Bearer " + api_key),
    }
    requests.post(
        scoring_url, data=json_text, headers=headers
    )  # Run the request twice since the first time takes a
    r = requests.post(
        scoring_url, data=json_text, headers=headers
    )  # little longer due to the loading of the model
    print(r)

    dupes_to_score = dupes_test.iloc[:5, num_dupes_to_score]

    text_data = list(map(text_to_json, dupes_to_score))  # Retrieve the text data
    for text in text_data:
        r = requests.post(scoring_url, data=text, headers=headers)
        print(r)


def get_dupes_test(directory: str) -> pd.DataFrame:
    """
    Load Duplicate Test CSV into Pandas Dataframe.

    :param directory: root directory of data_folder
    :return: pd.DataFrame from the loaded csv
    """
    dupes_test_path = directory + "/data_folder/dupes_test.tsv"
    dupes_test = pd.read_csv(dupes_test_path, sep="\t", encoding="latin1")
    return dupes_test

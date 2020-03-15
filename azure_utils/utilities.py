"""
ai-utilities - azure_utils/utilities.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import gzip
import json
import logging
import math
import os
import re

import pandas as pd
import requests
from azureml.core.authentication import (
    AuthenticationException,
    AzureCliAuthentication,
    InteractiveLoginAuthentication,
    ServicePrincipalAuthentication,
    AbstractAuthentication,
)
from dotenv import get_key


def check_login() -> bool:
    """

    :return:
    """
    try:
        os.popen("az account show")
        return True
    except OSError:
        return False


def read_csv_gz(url, **kwargs):
    """Load raw data from a .tsv.gz file into Pandas data frame."""
    dataframe = pd.read_csv(
        gzip.open(requests.get(url, stream=True).raw),
        sep="\t",
        encoding="utf8",
        **kwargs
    )
    return dataframe.set_index("Id")


def clean_text(text):
    """Remove embedded code chunks, HTML tags and links/URLs."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"<pre><code>.*?</code></pre>", "", text)
    text = re.sub(r"<a[^>]+>(.*)</a>", replace_link, text)
    return re.sub(r"<[^>]+>", "", text)


def replace_link(match):
    """

    :param match:
    :return:
    """
    if re.match(r"[a-z]+://", match.group(1)):
        return ""
    return match.group(1)


def round_sample(input_dataframe, frac=0.1, min_samples=1):
    """Sample X ensuring at least min samples are selected."""
    num_samples = max(min_samples, math.floor(len(input_dataframe) * frac))
    return input_dataframe.sample(num_samples)


def round_sample_strat(input_dataframe, strat, **kwargs):
    """Sample X ensuring at least min samples are selected."""
    return input_dataframe.groupby(strat).apply(round_sample, **kwargs)


def random_merge(
    dataframe_a, dataframe_b, number_to_merge=20, merge_col="AnswerId", key="key", n="n"
):
    """Pair all rows of A with 1 matching row on "on" and N-1 random rows from B"""
    assert key not in dataframe_a and key not in dataframe_b
    dataframe_a_copy = dataframe_a.copy()
    dataframe_a_copy[key] = dataframe_a[merge_col]
    dataframe_b_copy = dataframe_b.copy()
    dataframe_b_copy[key] = dataframe_b[merge_col]
    match = dataframe_a_copy.merge(dataframe_b_copy, on=key).drop(key, axis=1)
    match[n] = 0
    df_list = [match]
    for i in dataframe_a.index:
        dataframe_a_copy = dataframe_a.loc[[i]]
        dataframe_b_copy = dataframe_b[
            dataframe_b[merge_col] != dataframe_a_copy[merge_col].iloc[0]
        ].sample(number_to_merge - 1)
        dataframe_a_copy[key] = 1
        dataframe_b_copy[key] = 1
        z = dataframe_a_copy.merge(dataframe_b_copy, how="outer", on=key).drop(
            key, axis=1
        )
        z[n] = range(1, number_to_merge)
        df_list.append(z)
    return pd.concat(df_list, ignore_index=True)


def text_to_json(text):
    """

    :param text:
    :return:
    """
    return json.dumps({"input": "{0}".format(text)})


def get_auth(env_path: str) -> AbstractAuthentication:
    """

    :param env_path:
    :return:
    """
    logger = logging.getLogger(__name__)
    if get_key(env_path, "password") != "YOUR_SERVICE_PRINCIPAL_PASSWORD":
        logger.debug("Trying to create Workspace with Service Principal")
        aml_sp_password = get_key(env_path, "password")
        aml_sp_tennant_id = get_key(env_path, "tenant_id")
        aml_sp_username = get_key(env_path, "username")
        auth = ServicePrincipalAuthentication(
            tenant_id=aml_sp_tennant_id,
            service_principal_id=aml_sp_username,
            service_principal_password=aml_sp_password,
        )
    else:
        logger.debug("Trying to create Workspace with CLI Authentication")
        try:
            auth = AzureCliAuthentication()
            auth.get_authentication_header()
        except AuthenticationException:
            logger.debug("Trying to create Workspace with Interactive login")
            auth = InteractiveLoginAuthentication()

    return auth

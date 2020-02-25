"""
ai-utilities - utils.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from azure_utils import directory
from azure_utils.machine_learning.utils import load_configuration, get_or_create_workspace


def init_test_vars():
    """
    Load Common Vars for Testing

    :return: CONFIG, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME, WORKSPACE_REGION, workspace, SQL_DATASTORE_NAME,
    BLOB_DATASTORE_NAME
    :rtype: Union[Dict[Hashable, Any], list, None], str, str, str, str, Workspace, str, str
    """
    cfg = load_configuration(directory + "/../workspace_conf.yml")

    subscription_id = cfg['subscription_id']
    resource_group = cfg['resource_group']
    workspace_name = cfg['workspace_name']
    workspace_region = cfg['workspace_region']

    workspace = get_or_create_workspace(workspace_name, subscription_id, resource_group, workspace_region)

    sql_datastore_name = "ado_sql_datastore"
    blob_datastore_name = "ado_blob_datastore"

    return cfg, subscription_id, resource_group, workspace_name, workspace_region, workspace, sql_datastore_name, \
           blob_datastore_name

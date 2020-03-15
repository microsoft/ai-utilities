"""
AI-Utilities - storage.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from typing import Any, Tuple

from azure.mgmt.storage import StorageManagementClient
from azure.mgmt.storage.models import Kind, Sku, SkuName, StorageAccountCreateParameters

from azure_utils.azureml_tools.resource_group import create_resource_group


class StorageAccountCreateFailure(Exception):
    """Storage Account Create Failure Exception"""

    pass


def create_premium_storage(
    profile_credentials: object,
    subscription_id: str,
    location: str,
    resource_group_name: str,
    storage_name: str,
) -> Tuple[Any, dict]:
    """Create premium blob storage

    Args:
        profile_credentials : credentials from Azure login (see example below for details)
        subscription_id (str): subscription you wish to use
        location (str): location you wish the strage to be created in
        resource_group_name (str): the name of the resource group you want the storage to be created under
        storage_name (str): the name of the storage account

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]

    Example:
        >>> from azure.common.credentials import get_cli_profile
        >>> profile = get_cli_profile()
        >>> profile.set_active_subscription("YOUR-ACCOUNT")
        >>> cred, subscription_id, _ = profile.get_login_credentials()
        >>> storage = create_premium_storage(cred, subscription_id, "eastus", "testrg", "teststr", wait=False)
    """
    storage_client = StorageManagementClient(profile_credentials, subscription_id)
    create_resource_group(
        profile_credentials, subscription_id, location, resource_group_name
    )
    if not storage_client.storage_accounts.check_name_availability(
        storage_name
    ).name_available:
        storage_account = storage_client.storage_accounts.get_properties(
            resource_group_name, storage_name
        )
    else:
        storage_async_operation = storage_client.storage_accounts.create(
            resource_group_name,
            storage_name,
            StorageAccountCreateParameters(
                sku=Sku(name=SkuName.premium_lrs),
                kind=Kind.block_blob_storage,
                location="eastus",
            ),
        )
        storage_account = storage_async_operation.result()

    if "Succeeded" not in storage_account.provisioning_state:
        raise StorageAccountCreateFailure(
            f"Storage account not created successfully | State {storage_account.provisioning_state}"
        )

    storage_keys = storage_client.storage_accounts.list_keys(
        resource_group_name, storage_name
    )
    storage_keys = {v.key_name: v.value for v in storage_keys.keys}

    return storage_account, storage_keys

"""
azure-utils - configuration_validation.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import collections
import json
import os
import re
from enum import Enum
from typing import Optional

from azure.common.client_factory import get_client_from_cli_profile
from azure.mgmt.resource import ResourceManagementClient
from msrestazure.azure_exceptions import CloudError

azure_with_az_login = "Log into azure with 'az login'"


class ValidationType(Enum):
    """
    Enumerator used to look up validation rules. User passes in name
    and enumerator is looked up.
    """

    subscription_id = "subscription_id"
    resource_group = "resource_group"
    workspace_name = "workspace_name"
    storage_account = "storage_account"

    @classmethod
    def has_value(cls, value):
        """
        Check if ENUM has value

        :param value:
        :return:
        """
        return value in cls._value2member_map_


class ValidationResult(Enum):
    """ Enumerator used to Identify validation results """

    success = "PASSED"
    warning = "WARNING"
    failure = "FAILED"


# Result returned from Validation.validateInput()
#
# type - Named type passed in
# value - Value passed in
# status - bool indicating success or failure
# reason - Detailed explanation of what happened.
validation_result = collections.namedtuple(
    "validation_result", "type value status reason"
)


class ResultsGenerator:
    """ Collection of results for configurations """

    NAMES_LINK = "https://docs.microsoft.com/azure/azure-resource-manager/management/resource-name-rules"

    @staticmethod
    def create_length_failure(type_name, value, length) -> validation_result:
        """
        Failure due to Field Length

        :param type_name: String representing the field type being validated and is a value to one of the
        ValidationType enums. If not a valid value, no checks are performed.
        :param value: Value to validate.
        :param length: Length to validate
        :return: Returns a VALIDATION_RESULT failure object
        """
        return validation_result(
            type_name,
            value,
            ValidationResult.failure,
            "Field failed length validation: {} \n    See: {}".format(
                length, ResultsGenerator.NAMES_LINK
            ),
        )

    @staticmethod
    def create_content_failure(type_name, value, content) -> validation_result:
        """
        Failure due to Content Validation

        :param type_name: String representing the field type being validated and is a value to one of the
        ValidationType enums. If not a valid value, no checks are performed.
        :param value: Value to validate.
        :param content: Content to validate against
        :return: Returns a VALIDATION_RESULT failure object
        """
        return validation_result(
            type_name,
            value,
            ValidationResult.failure,
            "Field failed content validation by containing one or more of the following : {} \n  "
            "  See: {}".format(content, ResultsGenerator.NAMES_LINK),
        )

    @staticmethod
    def create_success(type_name, value, content) -> validation_result:
        """
        Create Successful Result

        :param type_name: String representing the field type being validated and is a value to one of the
        ValidationType enums. If not a valid value, no checks are performed.
        :param value: Value to validate.
        :param content: Content that succeeded validate
        :return: Returns a VALIDATION_RESULT success object
        """
        return validation_result(type_name, value, ValidationResult.success, content)

    @staticmethod
    def create_failure(type_name, value, content):
        """
        Create Failure Result

        :param type_name: String representing the field type being validated and is a value to one of the
        ValidationType enums. If not a valid value, no checks are performed.
        :param value: Value to validate.
        :param content: Content that failed validate
        :return: Returns a VALIDATION_RESULT failure object
        """
        return validation_result(type_name, value, ValidationResult.failure, content)

    @staticmethod
    def create_warning(type_name, value, content) -> validation_result:
        """
        Create Warning Result

        :param type_name: String representing the field type being validated and is a value to one of the
        ValidationType enums. If not a valid value, no checks are performed.
        :param value: Value to validate.
        :param content: Content that passed validate with warnings
        :return: Returns a VALIDATION_RESULT warning object
        """
        return validation_result(type_name, value, ValidationResult.warning, content)

    @staticmethod
    def create_generic_format_failure(type_name, value) -> validation_result:
        """
        Create Warning Result

        :param type_name: String representing the field type being validated and is a value to one of the
        ValidationType enums. If not a valid value, no checks are performed.
        :param value: Value to validate.
        :return: Returns a generic VALIDATION_RESULT failure object
        """
        return validation_result(
            type_name,
            value,
            ValidationResult.failure,
            "See: {}".format(ResultsGenerator.NAMES_LINK),
        )


class Validation:
    """
    Validation class for inputs based on types. Callers instantiate this class and
    then use the validateInput() routine to get results.

    Unknown types are automatically passed straight through with a success.
    """

    # Restrictions to be mapped internally in self.type_restrictions
    # length          - If not none, it's a list with two values [low,high] indicating
    #                   acceptable name lengths.
    # invalid_charset - If not none, a list of invalid characters for a given field.
    # custom_validator- If not none, a routine to call if length and content checks pass.

    FIELD_NOT_RECOGNIZED = "Field not recognized/validated."

    def __init__(self, default_field_value: str = "<>"):
        """
        Set up restrictions based on supported types. To add a new type field to validate:
        1. Extend the enum class ValidationType
        2. Add an entry into this dictionary.

        :param default_field_value: The default value for a field, default = <>
        """
        validation_restrictions = collections.namedtuple(
            "validation_restrictions",
            ["length", "regex_pattern", "invalid_charset", "custom_validator"],
        )
        self.type_restrictions = {
            ValidationType.subscription_id: validation_restrictions(
                None, False, "<>", self._validate_subscription
            ),
            ValidationType.resource_group: validation_restrictions(
                [1, 90], True, r"^[-\w\._\(\)]+$", self._validate_resource_group
            ),
            ValidationType.workspace_name: validation_restrictions(
                [1, 260], False, r"<>*%&:?+/\\", None
            ),
            ValidationType.storage_account: validation_restrictions(
                [1, 260], True, r"^[a-zA-Z0-9_\-]+$", None
            ),
        }

        # Get names of fields validated for checks by user.
        self.default_field_value = default_field_value
        self.validated_fields = [x.name for x in self.type_restrictions]

        # Used if subscription is validated, custom validation routine
        self.current_subscription = None

    def is_field_valid(self, field_name: str) -> bool:
        """
        Check if field name is valid.

        :param field_name: Field name to Test
        :return: `bool`
        """
        return field_name in self.validated_fields

    def validate_input(self, type_name: str, value: str) -> validation_result:
        """
        Validates the 'value' passed in for a given 'type_name'. If the type name is not
        a valid field in the enum class ValidationType, the check immediately passes.

        :param type_name: String representing the field type being validated and is a value to one
                          of the ValidationType enums. If not a valid value, no checks are performed.
        :param value: Value to validate.
        :return: Validation Results
        """
        return_result = None

        validation_type = self._get_validation_type(type_name)
        if validation_type:
            return_result = self.check_valid_type(
                return_result, type_name, validation_type, value
            )
        else:
            if value == self.default_field_value:
                return_result = ResultsGenerator.create_warning(
                    type_name, value, "Default value in field may cause problems."
                )
            else:
                return_result = ResultsGenerator.create_warning(
                    type_name, value, Validation.FIELD_NOT_RECOGNIZED
                )

        # If passed through to here with no return_result, all tests passed.

        if not return_result:
            return_result = ResultsGenerator.create_success(type_name, value, "")

        return return_result

    def check_valid_type(self, return_result, type_name, validation_type, value):
        """

        :param return_result:
        :param type_name:
        :param validation_type:
        :param value:
        :return:
        """
        # Type is valid, perform the following checks
        #
        # 1. Validate length requirements, if provided
        # 2. Validate content against invalid characters, if provided
        # 3. Custom validation, if provided
        if not self._validate_length(self.type_restrictions[validation_type], value):
            return_result = ResultsGenerator.create_length_failure(
                type_name, value, self.type_restrictions[validation_type].length
            )
        if not return_result and not self._validate_content(
            self.type_restrictions[validation_type], value
        ):
            return_result = ResultsGenerator.create_content_failure(
                type_name,
                value,
                self.type_restrictions[validation_type].invalid_charset,
            )
        if (
            not return_result
            and self.type_restrictions[validation_type].custom_validator
        ):
            return_result = self.type_restrictions[validation_type].custom_validator(
                type_name, value
            )
        return return_result

    @staticmethod
    def dump_validation_result(result: validation_result):
        """
        Print out validation results

        :param result: result to print
        """
        # type value status reason
        print("{} - {} - {}".format(result.status.value, result.type, result.value))
        if result.reason:
            print("  {}".format(result.reason))

    # Methods below this point are purely used to enable validateInput or any supporting
    # customized validations.

    @staticmethod
    def _validate_length(validation_restriction, value: str) -> bool:
        """
        Validates the length of the field IF
        1. There exists a valid VALIDATION_RESTRICTION
        2. That restriction has a 'length' list associated with it

        :param validation_restriction: Set of restrictions to validate length against
        :param value: value to test
        :return: `bool` result of length validation check
        """
        return_value = True
        if validation_restriction.length:
            return_value = (len(value) >= validation_restriction.length[0]) and (
                len(value) <= validation_restriction.length[1]
            )
        return return_value

    @staticmethod
    def _validate_content(validation_restriction, value: str) -> bool:
        """
        Validates the contents of the field IF
        1. There exists a valid VALIDATION_RESTRICTION
        2. That restriction has an invalid_charset


        One of two checks occurs based on the state of regex_pattern.
        1. True : Use regex looking for an EXACT match
        2. False : Compare all chars in the invalid set ensuring they don't appear in value

        :param validation_restriction: Set of restrictions to validate against
        :param value: value to test
        :return: `bool` result of length validation check
        """
        return_value = True
        if validation_restriction.invalid_charset:
            if validation_restriction.regex_pattern:
                requirement = re.compile(validation_restriction.invalid_charset)
                return_value = requirement.match(value) is not None
            else:
                for char in validation_restriction.invalid_charset:
                    if char in value:
                        return_value = False
                        break
        return return_value

    @staticmethod
    def _get_validation_type(type_name: str) -> Optional[ValidationType]:
        """
        Get the Validation Type from a type name

        :param type_name: Type name to look up :class:`ValidationType`
        :return: :class:`ValidationType` for input type name
        """
        if ValidationType.has_value(type_name):
            return ValidationType(type_name)
        return None

    # Custom validators : Subscription

    @staticmethod
    def _get_data_as_json(command: str) -> dict:
        """
        Run CLI Command and get output as JSON

        :param command: CLI Command to execute
        :return: JSON of output
        """
        account_stream = os.popen(command)
        command_output = account_stream.read().replace("[0m", "")
        account_stream.close()
        if not bool(command_output):
            return {}
        return json.loads(command_output)

    def _get_current_subscription(self) -> str:
        """
        Get Current Subscription from AZ CLI

        :return: Azure Subscription ID
        """
        try:
            result = self._get_data_as_json("az account show")
        except OSError:
            raise Exception("Try and run `az login`")

        if result:
            self.current_subscription = result["id"]
        return self.current_subscription

    def _validate_subscription(self, type_name, sub_id) -> validation_result:
        """
        Check that the current subscription is the subscription in the configuration

        :param type_name: String representing the field type being validated and is a value to one of the
        ValidationType enums. If not a valid value, no checks are performed.
        :param sub_id: Azure Subscription ID
        :return: Validation Results
        """
        return_result = ResultsGenerator.create_success(
            type_name, sub_id, azure_with_az_login
        )

        current_sub = self._get_current_subscription()
        if not current_sub:
            return_result = ResultsGenerator.create_failure(
                type_name, sub_id, azure_with_az_login
            )

        if not return_result and sub_id != current_sub:
            return_result = ResultsGenerator.create_warning(
                type_name,
                sub_id,
                "Subscription {} is not your current sub {}. Use 'az "
                "account set -s <subid>'".format(sub_id, current_sub),
            )
        return return_result

    def _validate_resource_group(self, type_name, group_name) -> validation_result:
        """
        Validate Resource Group

        :param type_name: String representing the field type being validated and is a value to one of the
        ValidationType enums. If not a valid value, no checks are performed.
        :param group_name: Resource Group to check.
        :return: Validation Results
        """
        return_result = ResultsGenerator.create_success(
            type_name, group_name, azure_with_az_login
        )

        try:
            current_sub = self._get_current_subscription()
            if not current_sub:
                return_result = ResultsGenerator.create_failure(
                    type_name, group_name, azure_with_az_login
                )
        except OSError:
            return_result = ResultsGenerator.create_failure(
                type_name, group_name, azure_with_az_login
            )
        if not return_result:
            rmc_client = get_client_from_cli_profile(ResourceManagementClient)
            try:
                res = rmc_client.resource_groups.check_existence(group_name)
                if res:
                    return_result = ResultsGenerator.create_warning(
                        type_name, group_name, "Resource Group Exists"
                    )
            except CloudError:
                return_result = ResultsGenerator.create_generic_format_failure(
                    type_name, group_name
                )

        return return_result

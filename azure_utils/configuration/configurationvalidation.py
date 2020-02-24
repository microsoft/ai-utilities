import collections
import typing
import os
import json
import re
from enum import Enum
from azure.common.client_factory import get_client_from_cli_profile
from azure.mgmt.resource import ResourceManagementClient
from msrest.exceptions import ValidationError

class ValidationType(Enum):
    """
        Enumerator used to look up validation rules. User passes in name
        and enumerator is looked up.
    """
    subscription_id = "subscription_id"
    resource_group = "resource_group"
    workspace_name = "workspace_name"
    storage_account = "storage_account"

class ValidationResult(Enum):
    """
        Enumerator used to Identify validation results
    """
    success = "PASSED"
    warning = "WARNING"
    failure = "FAILED"

'''
    Result returned from Validation.validateInput()

    type - Named type passed in
    value - Value passed in
    status - bool indicating success or failure
    reason - Detailed explaination of what happened.
'''
VALIDATION_RESULT = collections.namedtuple('validation_result', 'type value status reason')
class ResultsGenerator:
    NAMES_LINK = "https://docs.microsoft.com/azure/azure-resource-manager/management/resource-name-rules"

    @staticmethod
    def createLengthFailure(type_name, value, length):
        return VALIDATION_RESULT(
            type_name,
            value,
            ValidationResult.failure, 
            "Field failed length validation : {} \n    See: {}".format(length,ResultsGenerator.NAMES_LINK))

    @staticmethod
    def createContentFailure(type_name, value, content):
        return VALIDATION_RESULT(
            type_name,
            value,
            ValidationResult.failure, 
            "Field failed content validation by containing one or more of the following : {} \n    See: {}".format(content, ResultsGenerator.NAMES_LINK))

    @staticmethod
    def createSuccess(type_name, value, content):
        return VALIDATION_RESULT(
            type_name,
            value,
            ValidationResult.success, 
            content)

    @staticmethod
    def createFailure(type_name, value, content):
        return VALIDATION_RESULT(
            type_name,
            value,
            ValidationResult.failure, 
            content)

    @staticmethod
    def createWarning(type_name, value, content):
        return VALIDATION_RESULT(
            type_name,
            value,
            ValidationResult.warning, 
            content)

    @staticmethod
    def createGenericFormatFailure(type_name, value):
        return VALIDATION_RESULT(
            type_name,
            value,
            ValidationResult.failure, 
            "See: {}".format(ResultsGenerator.NAMES_LINK))

class Validation:
    """
        Validation class for inputs based on types. Callers instantiate this class and 
        then use the validateInput() routine to get results. 

        Unknown types are automatically passed straight through with a success.
    """

    '''
        Restrictions to be mapped internally in self.type_restrictions
        length          - If not none, it's a list with two values [low,high] indicating 
                          acceptable name lengths.
        invalid_charset - If not none, a list of invalid characters for a given field.
        custom_validator- If not none, a routine to call if length and content checks pass.
    '''
    FIELD_NOT_RECOTNIZED = "Field not recognized/validated."
    VALIDATION_RESTRICTIONS = collections.namedtuple('validation_restrictions', 'length regex_pattern invalid_charset custom_validator')

    def __init__(self, default_field_value : str = '<>'):
        '''
            Set up restrictions based on supported types. To add a new type field to validate:
            1. Extend the enum class ValidationType
            2. Add an entry into this dictionary.
        '''
        self.type_restrictions = {
            
                ValidationType.subscription_id : Validation.VALIDATION_RESTRICTIONS(None, False, "<>", self._validateSubscription),
                ValidationType.resource_group : Validation.VALIDATION_RESTRICTIONS([1,90], True, "^[-\w\._\(\)]+$", self._validateResoureGroup),
                ValidationType.workspace_name : Validation.VALIDATION_RESTRICTIONS([1,260],False, "<>*%&:?+/\\", None),
                ValidationType.storage_account : Validation.VALIDATION_RESTRICTIONS([1,260],True, "^[a-zA-Z0-9_\-]+$", None)
            }

        '''
            Get names of fields validated for checks by user.
        '''
        self.default_field_value = default_field_value
        self.validated_fields = [x.name for x in self.type_restrictions.keys()]

        '''
            Used if subscripiton is validated, custom validation routine 
        '''
        self.current_subscription = None

    def isFieldValid(self, field_name : str) -> bool:
        return field_name in self.validated_fields

    def validateInput(self, type_name : str, value : str) -> VALIDATION_RESTRICTIONS:
        """
            Validates the 'value' passed in for a given 'type_name'. If the type name is not
            a valid field in the enum class ValidationType, the check immediately passes.

            type_name = String representing the field type being validated and is a value to one
                        of the ValdationType enums. If not a valid value, no checks are performed.
            value     = Value to validate.
        """
        return_result = None

        validation_type = self._getValidationType(type_name)
        if validation_type:
            '''
                Type is valid, perform the following checks

                1. Validate length requirements, if provided
                2. Validate content against invalid characters, if provided
                3. Custom validation, if provided
            '''
            if not self._validateLength(self.type_restrictions[validation_type], value):
                return_result = ResultsGenerator.createLengthFailure(
                        type_name, value, self.type_restrictions[validation_type].length
                        )

            if not return_result:
                if not self._validateContent(self.type_restrictions[validation_type], value):
                    return_result = ResultsGenerator.createContentFailure(
                        type_name,value,self.type_restrictions[validation_type].invalid_charset
                        )

            if not return_result:
                if self.type_restrictions[validation_type].custom_validator:
                    return_result = self.type_restrictions[validation_type].custom_validator(type_name, value)
        else:
            if value == self.default_field_value:
                return_result = ResultsGenerator.createWarning(type_name,value,"Default value in field may cause problems.")
            else:
                return_result = ResultsGenerator.createWarning(type_name,value,Validation.FIELD_NOT_RECOTNIZED)

        '''
            If passed through to here with no return_result, all tests passed.
        '''
        if not return_result:
            return_result = ResultsGenerator.createSuccess(type_name,value,"")

        return return_result

    @staticmethod
    def dumpValidationResult(result : VALIDATION_RESULT) -> None:
        # type value status reason
        print("{} - {} - {}".format(result.status.value, result.type, result.value))
        if result.reason:
            print("  {}".format(result.reason))

    '''
        Methods below this point are purely used to enable validateInput or any supporting
        customized validations.
    '''
    def _validateLength(self, validation_restriction : VALIDATION_RESTRICTIONS, value : str) -> bool:
        """
            Validates the length of the field IF
            1. There exists a valid VALIDATION_RESTRICTION 
            2. That restriction has a 'length' list associated with it
        """
        return_value = True
        if validation_restriction.length:
            return_value = (len(value) >= validation_restriction.length[0]) and (len(value) <= validation_restriction.length[1] )  
        return return_value

    def _validateContent(self, validation_restriction : VALIDATION_RESTRICTIONS, value : str) -> bool:
        """
            Validates the contents of the field IF
            1. There exists a valid VALIDATION_RESTRICTION 
            2. That restriction has an invalid_charset


            One of two checks occurs based on the state of regex_pattern. 
            1. True : Use regex looking for an EXACT match
            2. False : Compare all chars in the invalid set ensuring they don't appear in value
        """
        return_value = True
        if validation_restriction.invalid_charset:
            if validation_restriction.regex_pattern :
                requirement = re.compile(validation_restriction.invalid_charset)
                return_value = (requirement.match(value) != None)
            else :
                for char in validation_restriction.invalid_charset:
                    if char in value:
                        return_value = False
                        break
        return return_value

    def _getValidationType(self, type_name : str) -> ValidationType:
        return_validation_type = None
        try:
            return_validation_type = ValidationType(type_name) 
        except ValueError as ex:
            pass
        return return_validation_type

    '''
        Custom validators : Subscription
    '''
    def _getDataAsJson(self, command):
        account_stream = os.popen(command)
        command_output = account_stream.read()

        if len(command_output) == 0:
                return None

        return json.loads(command_output)

    def _getCurrentSubscription(self):
        if self.current_subscription:
            return self.current_subscription

        result = self._getDataAsJson("az account show")
        if result:
            self.current_subscription = result['id']
        return self.current_subscription

    def _validateSubscription(self, type_name, sub_id):
        return_result = None

        current_sub = self._getCurrentSubscription()
        if not current_sub:
            return_result = ResultsGenerator.createFailure(type_name, sub_id, "Log into azure with 'az login'")

        if not return_result:
            if sub_id != current_sub:
                return_result = ResultsGenerator.createWarning(type_name, sub_id, "Subscripton {} is not your current sub {}. Use 'az account set -s <subid>'".format(sub_id, current_sub))

        return return_result

    '''
        Custom validators : resource group
    '''
    def _validateResoureGroup(self, type_name, group_name):
        return_result = None

        current_sub = self._getCurrentSubscription()
        if not current_sub :
            return_result = ResultsGenerator.createFailure(type_name, group_name, "Log into azure with 'az login'")

        if not return_result:
            rmc_client = get_client_from_cli_profile(ResourceManagementClient)
            try:
                res = rmc_client.resource_groups.check_existence(group_name)
                if res:
                    return_result = ResultsGenerator.createWarning(type_name, group_name,"Resource Group Exists")
            except Exception as ex:
                    return_result = ResultsGenerator.createGenericFormatFailure(type_name, group_name)

        return return_result

if __name__ == '__main__':
    print("MAIN")
    v = Validation()

    r = v.validateInput("subscription_id", "edf507a2-6235-46c5-b560-fd463ba2e771")
    Validation.dumpValidationResult(r)
    r = v.validateInput("foo", "asbas;klj;ijer;kasdf")
    Validation.dumpValidationResult(r)
    r = v.validateInput("workspace_name", "<mygroup>")
    Validation.dumpValidationResult(r)
    r = v.validateInput("resource_group", "dg-ml-cmk-eastus-acr")
    Validation.dumpValidationResult(r)

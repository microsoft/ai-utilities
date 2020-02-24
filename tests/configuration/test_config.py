"""
ai-utilities - test_config.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import os

from azure_utils.configuration.project_configuration import ProjectConfiguration, find_file


def test_config():
    new_config_file = "./testconfiguration.yml"
    project_name = "Test Project"

    # Make sure file doesn't exist
    remove_config_file(new_config_file)

    # Create a new one with a specific name and two settings
    proj_config = ProjectConfiguration(new_config_file)
    proj_config.set_project_name(project_name)
    proj_config.add_setting("sub_id", "Your Azure Subscription", "my_sub")
    proj_config.add_setting("workspace", "Your Azure ML Workspace", "my_ws")

    assert proj_config.project_name() == project_name
    assert proj_config.get_value('sub_id') == 'my_sub'
    assert proj_config.get_value('workspace') == 'my_ws'
    assert len(proj_config.get_settings()) == 2

    # Save it and ensure the file exists
    proj_config.save_configuration()
    assert os.path.isfile(new_config_file)

    # Load it and check what we have
    proj_config = ProjectConfiguration(new_config_file)
    assert proj_config.project_name() == project_name
    assert proj_config.get_value('sub_id') == 'my_sub'
    assert proj_config.get_value('workspace') == 'my_ws'
    assert len(proj_config.get_settings()) == 2

    # Change a setting and test we get the right value
    proj_config.set_value('sub_id', 'new_sub')
    assert proj_config.get_value('sub_id') == 'new_sub'

    remove_config_file(new_config_file)


def test_find_file():
    file1 = "this_is_file.txt"
    open(file1, "w+")
    found, path = find_file(file1)
    assert found
    os.remove(file1)

    found, path = find_file("not_this_is_file.txt")
    assert not found

    file2 = "../this_is_higher_file.txt"
    open(file2, "w+")
    found, path = find_file(file2)
    assert found
    os.remove(file2)


def remove_config_file(conf_file: str):
    """
    Clean up configuration file

    :param conf_file: location of configuration file
    """
    if os.path.isfile(conf_file):
        os.remove(conf_file)

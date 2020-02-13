from setuptools import setup, find_packages

setup(
    name="Microsoft-AI-Azure-Utilities",	     # This is the name of your PyPI-package.
    version="0.2.0",		     # Update the version number for new releases
    description="Utility Functions for AI Solutions",
    author="Daniel Ciborowski & Daniel Grecoe",
    author_email="dciborow@microsoft.com",
    url="https://github.com/microsoft/AI-Utilities",
    license="MIT",
    packages=find_packages(),
    install_requires=["azure-cli", 'azureml-core', 'python-dotenv']
)

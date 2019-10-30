from setuptools import setup, find_packages

setup(
    name='StatisticsTracker',	     # This is the name of your PyPI-package.
    version='0.1',		     # Update the version number for new releases
    packages=find_packages(),
    install_requires=["azure-cli-core==2.0.63", "azure-storage-blob==1.3.1"]
)

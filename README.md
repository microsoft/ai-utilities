[![Azure Notebooks](https://notebooks.azure.com/launch.png)](https://notebooks.azure.com/import/gh/microsoft/AI-Utilities)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/microsoft/ai-utilities/blob/master/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
# AI Utilities Sample Project
This project is a sample showcasing how to create a common repository for utilities that can be used by other AI projects. 

# Working with Project
## Build Project

```bash
pip install -e .\AI-Utilities\
```

## Run Unit Tests with Conda
```bash
conda env create -f environment.yml
conda activate ai-utilities
python -m ipykernel install --prefix=[anaconda-root-dir]\envs\ai-utilities --name ai-utilities
pytest tests
```

Example Anaconda Root Dir
C:\Users\dcibo\Anaconda3

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.



## Load Configuration

```r

load_configuration("project.yml")

```

%%writefile register_model.py
```r

def register_model:

```

%%writefile score.py
```r

def init:
def run(request): response

```

aks_service = RealtimeScore.get_or_create('train.py','score.py')
# Register Model
# Get or Deploy Application Insights
# Get or Deploy AKS
# Create service with /get endpoint for health check, 201 = healthy else dead (get endpoint gets ping tested)
# Copy App Insights keys into base R Image
# Get or Deploy Image to AKS
# Create Application Insights Ping/Simulation = Enterprise Promise Monitoring

# Final - Script to run this notebook for CI/CD


# Sample R and Project First Time Setup Shell Script

[See Here](https://github.com/Azure/AzureAuth/blob/master/azure-pipelines.yml)
```shell script
apt install r-base-core
```

# Sample Shell Script
```shell script
Rscript "rmarkdown::render('sample_r.rmd')"
```
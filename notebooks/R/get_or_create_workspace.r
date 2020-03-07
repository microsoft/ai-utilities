# Install Azure ML SDK from CRAN
install.packages("azuremlsdk")

# Then, use `install_azureml()` to install the compiled code from the AzureML Python SDK.
azuremlsdk::install_azureml()

library(azuremlsdk)
library(yaml)

get_or_create_workspace <- function(yaml_file) {

  configuration <- read_yaml(yaml_file)
  subscription_id <- configuration[['settings']][[1]][[1]][[2]][['value']]
  resource_group <- configuration[['settings']][[2]][[1]][[2]][['value']]
  ws_name <- configuration[['settings']][[3]][[1]][[2]][['value']]
  location <- configuration[['settings']][[4]][[1]][[2]][['value']]

  if (ws_name %in% names(list_workspaces(subscription_id = subscription_id))) {
    ws <- get_workspace(name = ws_name,
                        subscription_id = subscription_id,
                        resource_group = resource_group)
  } else {
    ws <- create_workspace(name = ws_name,
                           subscription_id = subscription_id,
                           resource_group = resource_group,
                           location = location)
  }
  write_workspace_config(ws)
  return(ws)
}
myacr='azmlrtseastuae34fcc7'
location='East US'
myresourcegroup=''
myplanname=''
app-name=''
acrinstance='azmlrtseastuae34fcc7'
imagename='20200302061056'
username=''
imagetag='latest'
password=''

az storage account create \
  --name "dcibfunctionstorage" \
  --location 'East US' \
  --resource-group "azmlrts-eastus-stable" \
  --sku Standard_LRS

az appservice plan create \
  --name "dcibfuncplan" \
  --resource-group "azmlrts-eastus-stable" \
  --is-linux

az functionapp create \
  --resource-group "azmlrts-eastus-stable" \
  --name "azmlrts" \
  --plan "dcibfuncplan" \
  --deployment-container-image-name azmlrtseastuae34fcc7.azurecr.io/package:93985ed7a0ffc24ccd583b1a1785b0de33a76ea2636528c29403943597a2a038 \
  --storage-account "dcibfunctionstorage"

az functionapp config container set \
  --name "azmlrts" \
  --resource-group "azmlrts-eastus-stable" \
  --docker-custom-image-name azmlrtseastuae34fcc7.azurecr.io/package:20200302061056 \
  --docker-registry-server-url https://azmlrtseastuae34fcc7.azurecr.io

  \
  --docker-registry-server-user $username \
  --docker-registry-server-password $password
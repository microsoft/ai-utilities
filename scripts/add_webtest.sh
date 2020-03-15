az group deployment create \
    --name ping-test-ml-3 \
    --resource-group azmlrts-eastus-stable \
    --template-file webtest.json \
    --parameters @webtest.parameters.json
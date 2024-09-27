az aks create \
    --resource-group soft-skills-coach-rg \
    --name chatbot_cluster \
    --node-count 1 \
     --node-vm-size standard_d16pls_v5 \
    --enable-addons monitoring \
    --kubernetes-version 1.30.4 \
    --generate-ssh-keys

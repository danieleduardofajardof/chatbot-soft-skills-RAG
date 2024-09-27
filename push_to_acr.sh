#!/bin/bash

# Set variables
RESOURCE_GROUP="soft-skills-coach-rg"
REGISTRY_NAME="softskillsregistry"  # Azure Container Registry name
IMAGE_NAME="softskillsbot"  # Docker image name
LOCATION="eastus"  # Change as needed

# Create the resource group (if it doesn't exist)
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create the container registry (if it doesn't exist)
az acr create --resource-group $RESOURCE_GROUP --name $REGISTRY_NAME --sku Basic --location $LOCATION

# Navigate to the project directory (adjust the path as needed)
cd /home/your_username/your_project_directory  # Replace with your actual project path

# Build the Docker image
sudo docker build -t $IMAGE_NAME .

# Log in to ACR
az acr login --name $REGISTRY_NAME

# Tag the Docker image
sudo docker tag $IMAGE_NAME $REGISTRY_NAME.azurecr.io/$IMAGE_NAME:latest

# Push the Docker image
sudo docker push $REGISTRY_NAME.azurecr.io/$IMAGE_NAME:latest

# Verify the push
az acr repository list --name $REGISTRY_NAME --output table
az acr repository show-tags --name $REGISTRY_NAME --repository $IMAGE_NAME --output table

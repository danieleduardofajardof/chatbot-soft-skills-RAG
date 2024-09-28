# AI-Powered Soft Skills Coach Chatbot

## Overview

The AI-Powered Soft Skills Coach Chatbot is designed to assist users in developing their soft skills through interactive conversations. This project implements a Retrieval-Augmented Generation (RAG) system utilizing Azure services, Kubernetes for deployment, and OpenAI's models for natural language understanding and generation. The chatbot integrates with Slack for real-time communication and employs voice interaction capabilities, allowing users to interact via speech.

## Objectives

- **RAG System Implementation**: Build a RAG system to enhance the chatbot's responses using a vector database for storing conversation history.
- **Slack Integration**: Create a Slack bot that interacts with users and provides responses.
- **Voice Interaction**: Implement Azure's speech-to-text and text-to-speech services for voice interaction.
- **Deployment**: Deploy the chatbot on Azure Kubernetes Service (AKS).

## Technology Stack

- **Backend Framework**: FastAPI
- **Machine Learning Framework**: Transformers (Hugging Face)
- **Database**: Azure Cosmos DB (MongoDB API)
- **Containerization**: Docker
- **Orchestration**: Azure Kubernetes Service (AKS)
- **Speech Services**: Azure Cognitive Services for Speech
- **Messaging Platform**: Slack API
- **Programming Language**: Python

## Setup Instructions

### Prerequisites

- Azure account
- Azure CLI installed on your local machine
- Docker installed on your local machine
- Basic knowledge of FastAPI and Python

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/soft-skills-chatbot.git
cd soft-skills-chatbot
```

### Step 2: Set Up Environment Variables

Create a `.env` file in the project root and set the following environment variables:
$AKS_SECRET_NAME = <secret_name>
```plaintext
COSMOS_DB_CONNECTION_STRING=<your_cosmos_db_connection_string>
SLACK_BOT_TOKEN=<your_slack_bot_token>
AZURE_SPEECH_KEY=<your_azure_speech_key>
AZURE_REGION=<your_azure_region>
```
Create AKS secret from .env file:
```bash

kubectl create secret generic $AKS_SECRET_NAME 
kubectl create secret generic softskills-secrets --from-env-file=.env
```

### Step 3: Login to ACR 

Log in to Azure:

```bash
az login
```

Set your resource group and registry name:

```bash
RESOURCE_GROUP=<resource_group_name>
REGISTRY_NAME=<registry_name>
```

Login to ACR

```bash
 az acr login --name $REGISTRY_NAME
 az acr login --name softskillsregistry
```

### Step 4: Build and Push Docker Image to Azure Container Registry

Log in to Azure:

```bash
az login
```

Set your resource group and registry name:

```bash
RESOURCE_GROUP=<resource_group_name>
REGISTRY_NAME=<registry_name>
```

Push the image:

```bash

 az acr login --name softskillsregistry
docker buildx build --platform linux/arm64 -t softskillsregistry.azurecr.io/softskillsbot:latest --push .

```

### Step 5: Deploy to Azure Kubernetes Service (AKS)

1. **Create AKS Cluster**:

   ```bash
   az aks create --resource-group $RESOURCE_GROUP --name soft-skills-aks --node-count 1 --enable-addons monitoring --generate-ssh-keys
   ```

2. **Get AKS Credentials**:

   ```bash
   az aks get-credentials --resource-group $RESOURCE_GROUP --name soft-skills-aks
   ```

3. **Deploy the Application**:

   Create a Kubernetes deployment and service YAML file:

   ```yaml
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: soft-skills-chatbot
  spec:
    replicas: 1
    selector:
      matchLabels:
        app: soft-skills-chatbot
    template:
      metadata:
        labels:
          app: soft-skills-chatbot
      spec:
        containers:
        - name: soft-skills-chatbot
          image: softskillsregistry.azurecr.io/softskillsbot:latest
          ports:
          - containerPort: 80
          env:
          - name: SLACK_BOT_TOKEN
            valueFrom:
              secretKeyRef:
                name: softskills-secrets
                key: SLACK_BOT_TOKEN
          - name: COSMOS_DB_CONNECTION_STRING
            valueFrom:
              secretKeyRef:
                name: softskills-secrets
                key: COSMOS_DB_CONNECTION_STRING
  ---
  apiVersion: v1
  kind: Service
  metadata:
    name: soft-skills-chatbot
  spec:
    type: LoadBalancer
    ports:
    - port: 80
      targetPort: 80
    selector:
      app: soft-skills-chatbot

   ```
  Create a network-policy.yaml for the cluster:
   ```yaml
   apiVersion: networking.k8s.io/v1
    kind: NetworkPolicy
    metadata:
      name: allow-slack-api
      namespace: default
    spec:
      podSelector:
        matchLabels:
          app: soft-skills-chatbot
      policyTypes:
      - Egress
      egress:
      - to:
        - ipBlock:
            cidr: 0.0.0.0/0
        ports:
        - protocol: TCP
          port: 443

   ```
   Apply the configuration:

   ```bash
   kubectl apply -f deployment.yaml
   kubectl apply -f network-policy.yaml
   ```

### Step 6: Access the Chatbot

Retrieve the external IP of the service:

```bash
kubectl get services
```

Once you have the external IP, you can interact with your chatbot through Slack.

### Step 7: Set up Slack API
In the Slack API put the external IP with the relevant FastAPI endpoin, and verify is actually working, set up the option to receive API calls when a message is sent in a relvant channel.

## Step 8: Configure CI/CD
In order to configure github action a folder needs to be created '.github/workflows' with the 'ci-cd-pipeline.yaml'
```yaml
   name: CI/CD Pipeline for AKS

on:
  push:
    branches:
      - main  # Trigger pipeline on push to the main branch

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest

    steps:
      # Checkout the code from GitHub
      - name: Checkout code
        uses: actions/checkout@v2

      # Set up Docker Buildx for multi-platform builds (optional if needed)
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      # Log in to Azure using service principal credentials stored in GitHub Secrets
      - name: Log in to Azure
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}

      # Log in to Azure Container Registry (ACR)
      - name: Log in to ACR
        run: |
          az acr login --name softskillsregistry

      # Build and push the Docker image to ACR, tag it with the commit hash
      - name: Build and Push Docker Image
        run: |
          docker buildx build --platform linux/arm64 -t softskillsregistry.azurecr.io/softskillsbot:${{ github.sha }} --push .

      # Get credentials for the AKS cluster
      - name: Get AKS credentials
        run: |
          az aks get-credentials --resource-group soft-skills-coach-rg --name chatbot_cluster

      # Step 1: Explicitly delete the old deployment (optional)
      - name: Delete previous Kubernetes deployment
        run: |
          kubectl delete deployment soft-skills-chatbot || true

      # Step 2: Replace placeholder in deployment.yaml with the actual commit hash (dynamic image tag)
      - name: Update deployment.yaml with image tag
        run: |
          sed -i 's#__IMAGE_TAG__#${{ github.sha }}#g' deployment.yaml

      # Step 3: Apply the Kubernetes deployment with the updated image
      - name: Apply Kubernetes Deployment
        run: |
          kubectl apply -f deployment.yaml

      # Step 4: Optionally check the status of the rollout
      - name: Check Deployment Status
        run: |
          kubectl rollout status deployment/soft-skills-chatbot

   ```
   Also set the connection to AKS, by putting the Azure Subscription and other relevant informaton as ACR name

## Step 8: Test CI/CD
In order to test the CI/CD, you will need to make some change to the code and the push those to the repo
```bash
git add.
git commit -m "Some comments..."
git push
```
In order to avoid login everytime, add your SSH to the repo configuration for SSH Authentication.
## Implementation Details

The chatbot utilizes a RAG approach, retrieving relevant information from stored conversations to enhance its responses. Key functionalities include:

- **Voice Interaction**: Utilizing Azure's Cognitive Services, the bot converts speech to text for user input and text to speech for bot responses.
- **Logging and Monitoring**: Conversations are logged in Azure Cosmos DB, with monitoring for performance insights.
- **Slack Integration**: The bot responds to messages in Slack, facilitating user interaction.

## Known Limitations

- Speech recognition accuracy may vary based on user accents and clarity.
- Chatbot responses may not always align perfectly with user expectations.

## Potential Improvements

- Implement sentiment analysis to tailor responses based on user emotions.
- Expand conversational capabilities with more advanced NLP techniques.
- Enhance voice interaction with additional language support.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

Special thanks to the contributors of the libraries and tools used in this project, including Azure, Kubernetes, and Hugging Face Transformers.
# chatbot-soft-skills-RAG

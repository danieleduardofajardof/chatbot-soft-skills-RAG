# AI-Powered Soft Skills Coach Chatbot

# AI-Powered Soft Skills Coach Chatbot

## Overview

The AI-Powered Soft Skills Coach Chatbot is designed to assist users in developing their soft skills through interactive, AI-driven conversations. This project leverages a combination of Retrieval-Augmented Generation (RAG) techniques, advanced sentiment analysis, and Azure services to provide a highly personalized and adaptive learning experience. The chatbot integrates with Slack for real-time communication and supports voice interaction, allowing users to interact via speech. Additionally, the chatbot is designed to run efficiently on Azure Kubernetes Service (AKS) for scalable deployment.

## Objectives

- **RAG System Implementation**: Build a Retrieval-Augmented Generation (RAG) system to improve the chatbot's responses by combining the power of generative models with a vector database that stores conversation history and relevant documents. This enables the chatbot to retrieve and generate more accurate and context-aware responses.
  
- **HyDE Technique**: Implement Hypothetical Document Embeddings (HyDE) to further enhance the chatbot’s understanding and response generation by synthesizing hypothetical documents from user inputs and matching them with similar content in the database.
  
- **Sentiment Analysis**: Integrate sentiment analysis using OpenAI's GPT-3.5 models to detect and interpret the emotional tone of user inputs. Based on the detected sentiment (positive, neutral, or negative), the chatbot adapts its responses to be more empathetic, encouraging, or supportive, providing a more human-like interaction experience.

- **Slack Integration**: Deploy the chatbot as a Slack bot that interacts with users in real time, providing responses based on both text and audio inputs. The chatbot can engage in multi-turn conversations while maintaining context to provide meaningful and continuous dialogue.

- **Voice Interaction**: Enable voice interaction by integrating Azure's Speech-to-Text and Text-to-Speech services. This allows users to communicate with the bot via voice commands, and the bot responds with synthesized voice output, making the interaction more dynamic and accessible.

- **Deployment on AKS**: Ensure the chatbot is scalable and highly available by deploying it on Azure Kubernetes Service (AKS). This allows for efficient orchestration of containerized services and provides scalability to handle multiple user requests simultaneously.

- **Azure Blob Storage**: The chatbot uses Azure Blob Storage to store and retrieve audio files processed during voice interactions. The audio files are stored in a secure, scalable cloud storage, enabling quick access and retrieval of media files during interactions.

- **Cosmos DB**: A MongoDB-compatible Cosmos DB is used to store logs, user interactions, and conversation history. This allows for quick access and searchability of past interactions, enabling the chatbot to maintain context and improve future responses.

- **GitHub Actions for CI/CD**: Continuous Integration and Continuous Deployment (CI/CD) is implemented using GitHub Actions. This ensures automated deployment, testing, and monitoring of the chatbot application whenever updates are pushed to the GitHub repository, keeping the system robust and up-to-date.

## Key Features

- **Retrieval-Augmented Generation (RAG)**: The chatbot enhances its responses by combining real-time generation with information retrieval from a knowledge base, improving the relevance and accuracy of responses.
  
- **HyDE Technique**: The chatbot utilizes Hypothetical Document Embeddings to generate more insightful responses by synthesizing potential document matches to user queries.

- **Sentiment Analysis**: Using GPT-3.5, the chatbot can analyze the sentiment behind user messages, allowing it to provide more personalized and emotionally intelligent responses.

- **Slack Bot**: The chatbot integrates seamlessly with Slack, offering both text-based and voice-based interaction through the Slack platform.

- **Voice-to-Text and Text-to-Voice**: Leveraging Azure Cognitive Services, users can speak to the bot and receive voice responses, enhancing accessibility and user engagement.

- **Azure Blob Storage**: Audio files generated during voice interactions are stored and managed using Azure Blob Storage for secure and scalable cloud-based storage.

- **Cosmos DB**: The bot stores interaction logs and conversation history in Cosmos DB, enabling it to access previous conversations and maintain context.

- **Azure Kubernetes Service (AKS) Deployment**: The chatbot is deployed on AKS, ensuring a scalable, secure, and robust environment for handling large volumes of interactions.

- **GitHub Actions for CI/CD**: GitHub Actions automate the build, test, and deployment processes, ensuring smooth and continuous delivery of updates.

This project combines cutting-edge AI technologies with robust deployment strategies to create an adaptive and responsive soft skills coach that can improve users' communication and interpersonal skills in real time.

## Technology Stack

- **Backend Framework**: FastAPI
- **Machine Learning Framework**: Azure Open AI 
- **Database**: Azure Cosmos DB (MongoDB API)
- **Containerization**: Docker
- **Orchestration**: Azure Kubernetes Service (AKS)
- **Storage**: Azure Blob Storage Service
- **Speech Services**: Azure Cognitive Services for Speech
- **Messaging Platform**: Slack API
- **Programming Language**: Python
- **CI/CD**: Github Actions

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
        securityContext:
          runAsUser: 0            # Run all containers as this user
          runAsGroup: 0           # Run all containers in this group
          fsGroup: 0              # File system group for the pod
          seccompProfile:
            type: RuntimeDefault 
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
          - name: AZURE_OPENAI_API_KEY
            valueFrom:
              secretKeyRef:
                name: softskills-secrets
                key: AZURE_OPENAI_API_KEY
          - name: AZURE_OPENAI_ENDPOINT
            valueFrom:
              secretKeyRef:
                name: softskills-secrets
                key: AZURE_OPENAI_ENDPOINT
          - name: AZURE_REGION
            valueFrom:
              secretKeyRef:
                name: softskills-secrets
                key: AZURE_REGION


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

## Step 9: Inspect logs and run commands inside AKS
```bash
kubectl logs $(kubectl get pods --selector=app=soft-skills-chatbot -o jsonpath="{.items[0].metadata.name}")
kubectl exec -it $(kubectl get pods --selector=app=soft-skills-chatbot -o jsonpath="{.items[0].metadata.name}") -- /bin/sh
```


## Step 9: Update secfrets inside AKS
```bash
kubectl apply -f softskills-secrets.yaml 
```
Having a secrets.yaml like this:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: softskills-secrets
type: Opaque
stringData:  # stringData allows plain text; Kubernetes will automatically base64 encode the values.
  AZURE_OPENAI_API_KEY: <secret>
  AZURE_OPENAI_ENDPOINT: <secret>
  AZURE_REGION: "eastus"
  COSMOS_DB_CONNECTION_STRING: <secret>
  SLACK_BOT_TOKEN: <secret>
  AZURE_SPEECH_API_KEY: <secret>
  AZURE_STORAGE_CONNECTION_STRING: <secret>
  AZURE_STORAGE_CONTAINER_NAME: <secret>
```





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

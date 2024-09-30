#!/usr/bin/env python
import os
import logging
from fastapi import FastAPI, Request
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from pymongo import MongoClient
from datetime import datetime
from fastapi.responses import JSONResponse
import azure.cognitiveservices.speech as speechsdk
import requests
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from pydub import AudioSegment
import io

# Custom logging handler to store logs in CosmosDB
class CosmosDBHandler(logging.Handler):
    """Custom logging handler that stores logs in a MongoDB collection."""
    def emit(self, record):
        """
        Logs the record to MongoDB.
        
        Parameters:
        - record: logging.LogRecord - The log record to be stored.
        """
        log_entry = {
            "level": record.levelname,
            "message": record.getMessage(),
            "timestamp": datetime.utcnow().isoformat(),
            "module": record.module,
            "function": record.funcName,
            "line_no": record.lineno,
        }
        logs_collection.insert_one(log_entry)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(CosmosDBHandler())  # Add custom handler to store logs in CosmosDB

# MongoDB connection for logging
mongo_client = MongoClient(os.getenv("COSMOS_DB_CONNECTION_STRING"))
db = mongo_client['soft_skills_chatbot']
logs_collection = db['logs']

# Logging for environment variables
logger.info(f"AZURE_STORAGE_CONNECTION_STRING exists: {bool(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))}")
logger.info(f"AZURE_STORAGE_CONTAINER_NAME exists: {bool(os.getenv('AZURE_STORAGE_CONTAINER_NAME'))}")

# Blob Storage setup
try:
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
    logger.info("Successfully connected to Azure Blob Storage")
except Exception as e:
    logger.error(f"Failed to connect to Azure Blob Storage: {str(e)}")

container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
container_client = blob_service_client.get_container_client(container_name)

# Ensure the container exists
try:
    container_client.create_container()
except Exception as e:
    logger.info(f"Container already exists: {str(e)}")

# Azure OpenAI client initialization
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  # Key for GPT-3.5 model
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2023-03-15-preview"
)

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
async def root() -> dict:
    """
    Root route for basic health check.
    
    Returns:
    - dict: A dictionary containing a health check message.
    """
    return {"message": "This is the Soft Skills Chatbot API."}

@app.on_event("startup")
async def startup_event() -> None:
    """
    Startup event handler that logs directory permissions.
    
    Returns:
    - None
    """
    try:
        app_dir_stat = os.stat("/app")
        logger.info(f"Directory /app permissions: {oct(app_dir_stat.st_mode)}")
        logger.info(f"Owner UID: {app_dir_stat.st_uid}, GID: {app_dir_stat.st_gid}")
    except Exception as e:
        logger.error(f"Error checking /app directory permissions: {str(e)}")

# Initialize Slack client with the bot token
slack_client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

def speech_to_text(file_path: str) -> str:
    """
    Converts speech in an audio file to text using Azure Speech-to-Text.
    
    Parameters:
    - file_path: str - The path to the audio file.

    Returns:
    - str: The recognized speech as text or None if recognition fails.
    """
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_API_KEY"), region=os.getenv("AZURE_REGION"))
    audio_config = speechsdk.audio.AudioConfig(filename=file_path)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    result = recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        logger.info(f"Speech recognized: {result.text}")
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        logger.error("No speech could be recognized.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        logger.error(f"Speech Recognition canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            logger.error(f"Error details: {cancellation_details.error_details}")
    return None

def text_to_speech(response_text: str) -> str:
    """
    Converts text to speech using Azure Speech-to-Text and uploads it to Blob Storage.
    
    Parameters:
    - response_text: str - The text to be converted to speech.
    
    Returns:
    - str: The URL of the uploaded audio file in Blob Storage or None if the upload fails.
    """
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_API_KEY"), region=os.getenv("AZURE_REGION"))  
    audio_config = speechsdk.audio.AudioOutputConfig(filename="/tmp/response_audio.wav")  # Temporary file

    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    synthesizer.speak_text_async(response_text).get()

    # Upload the synthesized audio to Blob Storage
    blob_name = "response_audio.wav"
    blob_url = upload_to_blob("/tmp/response_audio.wav", blob_name)
    
    if blob_url:
        logger.info(f"Response audio uploaded to Blob Storage: {blob_url}")
        return blob_url
    else:
        logger.error("Failed to upload response audio to Blob Storage")
        return None

def convert_m4a_to_wav(input_file_path: str, output_file_path: str) -> None:
    """
    Converts an M4A audio file to WAV format.
    
    Parameters:
    - input_file_path: str - The path to the input M4A file.
    - output_file_path: str - The path to save the converted WAV file.
    
    Returns:
    - None
    """
    audio = AudioSegment.from_file(input_file_path, format="m4a")
    audio.export(output_file_path, format="wav")
    logger.info(f"Converted {input_file_path} to {output_file_path}")

def upload_to_blob(file_path: str, blob_name: str) -> str:
    """
    Uploads a file to Azure Blob Storage.
    
    Parameters:
    - file_path: str - The path to the file to be uploaded.
    - blob_name: str - The name of the blob in Azure Blob Storage.
    
    Returns:
    - str: The URL of the uploaded file in Blob Storage, or None if the upload fails.
    """
    try:
        blob_client = container_client.get_blob_client(blob_name)
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        logger.info(f"Uploaded {file_path} to Blob Storage as {blob_name}")
        return blob_client.url  # Return the Blob URL for access
    except Exception as e:
        logger.error(f"Failed to upload {file_path} to Blob Storage: {str(e)}")
        return None

def download_blob_to_local(blob_url: str, local_path: str) -> str:
    """
    Downloads the blob from Azure Blob Storage to a local path.
    
    Parameters:
    - blob_url: str - The URL of the blob to be downloaded.
    - local_path: str - The local path where the blob should be saved.

    Returns:
    - str: The local path where the blob was saved, or None if there was an error.
    """
    try:
        # Extract the blob name from the URL
        blob_name = blob_url.split("/")[-1]

        # Create a blob client for the specific blob
        blob_client = container_client.get_blob_client(blob_name)

        # Download the blob content to the local file system
        with open(local_path, "wb") as file:
            blob_data = blob_client.download_blob().readall()
            file.write(blob_data)

        logger.info(f"Downloaded blob to local path: {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Error downloading blob to local path: {e}")
        return None

def process_audio_file(file_url: str, token: str) -> str:
    """
    Processes an audio file from Slack: downloads, uploads to Blob, converts to WAV, and transcribes.
    
    Parameters:
    - file_url: str - The URL of the audio file to be processed.
    - token: str - The Slack authentication token for accessing the file.

    Returns:
    - str: The transcribed text or None if transcription fails.
    """
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    try:
        logger.info(f"Downloading file from Slack URL: {file_url}")
        response = requests.get(file_url, headers=headers, stream=True)
        if response.status_code == 200:
            logger.info(f"File download from Slack successful. Status code: {response.status_code}")

            # Save the raw response content to a BytesIO object
            file_in_memory = io.BytesIO(response.content)
            file_in_memory.seek(0)  # Reset pointer to the start of the file

            # Upload the .m4a file to Blob Storage directly
            blob_name_m4a = "received_audio.m4a"
            blob_client = container_client.get_blob_client(blob_name_m4a)
            blob_client.upload_blob(file_in_memory, overwrite=True)
            logger.info(f".m4a file uploaded directly to Blob Storage: {blob_client.url}")

            # Reset the file pointer again before converting
            file_in_memory.seek(0)

            # Convert the .m4a file to .wav in memory using pydub
            audio_data = AudioSegment.from_file(file_in_memory, format="m4a")
            wav_file_buffer = io.BytesIO()
            audio_data.export(wav_file_buffer, format="wav")
            wav_file_buffer.seek(0)  # Reset pointer to the start of the WAV file

            # Upload the .wav file to Blob Storage
            blob_name_wav = "converted_audio.wav"
            blob_client_wav = container_client.get_blob_client(blob_name_wav)
            blob_client_wav.upload_blob(wav_file_buffer, overwrite=True)
            logger.info(f".wav file uploaded directly to Blob Storage: {blob_client_wav.url}")

            # Transcribe the .wav file using Azure Speech-to-Text
            local_wav_file = download_blob_to_local(blob_client_wav.url, "/tmp/converted_audio.wav")
            if local_wav_file:
                transcribed_text = speech_to_text(local_wav_file)
                if transcribed_text:
                    logger.info(f"Transcribed text: {transcribed_text}")
                    return transcribed_text
                else:
                    logger.error("Failed to transcribe audio")
            else:
                logger.error("Failed to download and convert audio file")
        else:
            logger.error(f"Failed to download file from Slack. Status code: {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"Failed to process audio file: {str(e)}")
        return None


def generate_response(user_input: str) -> str:
    """
    Generates a chatbot response using Azure OpenAI GPT-3.5.
    
    Parameters:
    - user_input: str - The input text from the user.
    
    Returns:
    - str: The generated response or an error message if generation fails.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-35-turbo",  # Ensure this matches the deployment name in Azure for GPT-3.5
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7
        )
        bot_response = response.choices[0].message.content.strip()
        logger.info(f"Generated bot response: {bot_response}")
        return bot_response
    except Exception as e:
        logger.error(f"Error generating response from Azure OpenAI: {e}")
        return "I'm sorry, I'm having trouble generating a response right now."

def send_response_to_slack(channel: str, response: str, file_path: str = None) -> None:
    """
    Sends a response back to Slack, either as text or an audio file.
    
    Parameters:
    - channel: str - The Slack channel where the response should be sent.
    - response: str - The response text to be sent.
    - file_path: str, optional - The path to an audio file to be uploaded (default is None).
    
    Returns:
    - None
    """
    try:
        if file_path:
            # Download the file from Azure Blob Storage to a local path
            local_audio_file = download_blob_to_local(file_path, "/tmp/response_audio.wav")
            
            # Ensure the file was successfully downloaded
            if local_audio_file:
                logger.info(f"Uploading audio response to Slack: {local_audio_file}")
                with open(local_audio_file, "rb") as audio_file:
                    slack_client.files_upload_v2(channels=channel, file=audio_file, title="Response Audio")
            else:
                logger.error("Failed to download audio file for Slack upload.")
        else:
            slack_client.chat_postMessage(channel=channel, text=response)
    except SlackApiError as e:
        logger.error(f"Error sending message to Slack: {e.response.error}")
    except Exception as e:
        logger.error(f"Unexpected error sending message: {e}")

def log_conversation(user_id: str, user_input: str, bot_response: str) -> None:
    """
    Logs user and bot interactions to the database.
    
    Parameters:
    - user_id: str - The ID of the user who interacted with the bot.
    - user_input: str - The input provided by the user.
    - bot_response: str - The response generated by the bot.
    
    Returns:
    - None
    """
    log_entry = {
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "user_input": user_input,
        "bot_response": bot_response
    }
    logs_collection.insert_one(log_entry)
    
def generate_response_with_rag(user_input: str) -> str:
    # Retrieve relevant information from the database
    retrieved_info = retrieve_documents(user_input)
    
    # Combine the user input with the retrieved information
    combined_input = f"User Query: {user_input}\n\nRelevant Info: {retrieved_info}"
    
    # Call the OpenAI model with the combined input
    try:
        response = openai_client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": combined_input}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7
        )
        bot_response = response.choices[0].message.content.strip()
        logger.info(f"Generated bot response: {bot_response}")
        return bot_response
    except Exception as e:
        logger.error(f"Error generating response from Azure OpenAI: {e}")
        return "I'm sorry, I'm having trouble generating a response right now."

def retrieve_documents(user_input: str):
    """
    Retrieves documents from MongoDB that match the user input using a regular expression search.
    
    Parameters:
    - user_input: str - The input to search for within the document content.
    
    Returns:
    - str: The concatenated content of the matched documents.
    """
    try:
        query = {"content": {"$regex": user_input, "$options": "i"}}  # Case-insensitive regex search
        documents = logs_collection.find(query)

        if documents:
            return " ".join([doc['content'] for doc in documents])
        else:
            return "No relevant documents found."
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return "An error occurred while retrieving documents."


@app.post('/slack/events')
async def slack_events(req: Request) -> JSONResponse:
    """
    Slack event handler to process events such as messages or file shares.

    Parameters:
    - req: Request - The incoming request object from Slack containing the event data.
    
    Returns:
    - JSONResponse: A JSON response indicating success or failure.
    """
    data = await req.json()
    logger.info(f"Received event: {data}")

    if data.get("type") == "url_verification":
        challenge = data.get("challenge")
        return JSONResponse(status_code=200, content={"challenge": challenge})

    if 'event' in data:
        event = data['event']

        # Handle file share events
        
                    
        # Handle text message events
        if event.get('type') == 'message' and 'subtype' not in event:
            logger.info("Processing text message")
            user_input = event.get('text', '').strip()
            user_id = event.get('user', '')
            channel = event.get('channel', '')

            # Get the bot's user ID to prevent it from responding to itself
            auth_response = slack_client.auth_test()
            bot_user_id = auth_response['user_id']

            if user_input and user_id and channel and user_id != bot_user_id:
                bot_response = generate_response_with_rag(user_input)
                log_conversation(user_id, user_input, bot_response)
                send_response_to_slack(channel, bot_response)

        elif 'files' in event:
            logger.info("Files found in the event")
            for file in event.get('files'):
                user_id = event.get('user', '')
                auth_response = slack_client.auth_test()
                bot_user_id = auth_response['user_id']
                if user_id == bot_user_id:
                    logger.info("Skipping file event from the bot itself.")
                    return JSONResponse(status_code=200, content={"status": "skipped"})
                
                logger.info(f"File received: {file}")
                file_url = file.get('url_private')
                token = os.getenv("SLACK_BOT_TOKEN")
                try:
                    transcribed_text = process_audio_file(file_url, token)
                    if transcribed_text:
                        bot_response = generate_response(transcribed_text)
                        # Convert bot response to audio
                        audio_file_path = text_to_speech(bot_response)
                        logger.info(f"Generated audio file path: {audio_file_path}")
                        send_response_to_slack(event.get('channel'), bot_response, audio_file_path)
                    else:
                        send_response_to_slack(event.get('channel'), "Sorry, I couldn't understand the audio.")
               
                except Exception as e:
                    logger.error(f"Failed to process audio file: {str(e)}")

    return JSONResponse(status_code=200, content={"status": "success"})

@app.get("/healthz")
async def healthz() -> dict:
    """
    Health check endpoint.
    
    Returns:
    - dict: A dictionary indicating the service's health status.
    """
    return {"status": "healthy"}
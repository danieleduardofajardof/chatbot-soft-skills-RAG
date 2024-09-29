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
# Custom logging handler to store logs in CosmosDB
class CosmosDBHandler(logging.Handler):
    def emit(self, record):
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


logger.info(f"AZURE_STORAGE_CONNECTION_STRING exists: {bool(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))}")
logger.info(f"AZURE_STORAGE_CONTAINER_NAME exists: {bool(os.getenv('AZURE_STORAGE_CONTAINER_NAME'))}")


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


# MongoDB connection for logging
mongo_client = MongoClient(os.getenv("COSMOS_DB_CONNECTION_STRING"))
db = mongo_client['soft_skills_chatbot']
logs_collection = db['logs']


# Log environment variables for debugging (only log that they exist, no values)
logger.info(f"AZURE_OPENAI_API_KEY exists: {bool(os.getenv('AZURE_OPENAI_API_KEY'))}")
logger.info(f"AZURE_SPEECH_API_KEY exists: {bool(os.getenv('AZURE_SPEECH_API_KEY'))}")
logger.info(f"AZURE_OPENAI_ENDPOINT exists: {bool(os.getenv('AZURE_OPENAI_ENDPOINT'))}")

# Initialize the OpenAI client for GPT-3.5 (Azure OpenAI)
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  # Key for GPT-3.5 model
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2023-03-15-preview"
)

# Initialize FastAPI app
app = FastAPI()

# Root route for health check or basic information
@app.get("/")
async def root():
    return {"message": "This is the Soft Skills Chatbot API."}
@app.on_event("startup")
async def startup_event():
    try:
        # Log file permissions for /app directory
        app_dir_stat = os.stat("/app")
        logger.info(f"Directory /app permissions: {oct(app_dir_stat.st_mode)}")
        logger.info(f"Owner UID: {app_dir_stat.st_uid}, GID: {app_dir_stat.st_gid}")
    except Exception as e:
        logger.error(f"Error checking /app directory permissions: {str(e)}")

# Initialize Slack client with the bot token
slack_client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

# Azure Speech to Text using AZURE_SPEECH_API_KEY
def speech_to_text(file_path):
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

def text_to_speech(response_text):
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

# Process audio files and convert to text
from pydub import AudioSegment
import os

def convert_m4a_to_wav(input_file_path, output_file_path):
    """
    Converts an M4A audio file to WAV format.
    :param input_file_path: Path to the input M4A file.
    :param output_file_path: Path to save the converted WAV file.
    """
    audio = AudioSegment.from_file(input_file_path, format="m4a")
    audio.export(output_file_path, format="wav")
    logger.info(f"Converted {input_file_path} to {output_file_path}")

# Example usage in process_audio_file function
def upload_to_blob(file_path, blob_name):
    try:
        # Create a blob client
        blob_client = container_client.get_blob_client(blob_name)
        
        # Upload the file to the blob
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        logger.info(f"Uploaded {file_path} to Blob Storage as {blob_name}")
        return blob_client.url  # Return the Blob URL for access
    except Exception as e:
        logger.error(f"Failed to upload {file_path} to Blob Storage: {str(e)}")
        return None

def process_audio_file(file_url, token):
    headers = {
        "Authorization": f"Bearer {token}"
    }
    response = requests.get(file_url, headers=headers)
    file_path = "/tmp/received_audio.m4a"  # Temporary local file path
    
    try:
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            logger.info(f"Received and saved audio file locally to {file_path}, file size: {file_size} bytes")
            
            # Upload the .m4a file to Blob Storage
            blob_name_m4a = "received_audio.m4a"
            blob_url_m4a = upload_to_blob(file_path, blob_name_m4a)
            
            # Convert .m4a to .wav
            wav_file_path = "/tmp/converted_audio.wav"
            convert_m4a_to_wav(file_path, wav_file_path)

            # Upload the .wav file to Blob Storage
            blob_name_wav = "converted_audio.wav"
            blob_url_wav = upload_to_blob(wav_file_path, blob_name_wav)

            # Use Azure Speech-to-Text on the uploaded .wav file
            transcribed_text = speech_to_text(wav_file_path)
            
            if transcribed_text:
                logger.info(f"Transcribed Text: {transcribed_text}")
                return transcribed_text
            else:
                logger.error("Failed to transcribe the audio.")
                return None
        else:
            logger.error(f"File {file_path} not found after download.")
            return None
    except Exception as e:
        logger.error(f"Failed to process audio file: {str(e)}")
        return None


# Generate a response using Azure OpenAI's GPT model (GPT-3.5)
def generate_response(user_input):
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
        # Log the API response for debugging
        bot_response = response.choices[0].message.content.strip()
        logger.info(f"Generated bot response: {bot_response}")
        return bot_response
    except Exception as e:
        logger.error(f"Error generating response from Azure OpenAI: {e}")
        return "I'm sorry, I'm having trouble generating a response right now."

# Send response back to Slack
def send_response_to_slack(channel, response, file_path=None):
    try:
        if file_path:
            logger.info(f"Uploading audio response to Slack: {file_path}")
            slack_client.files_upload(channels=channel, file=file_path, title="Response Audio")
        else:
            slack_client.chat_postMessage(channel=channel, text=response)
    except SlackApiError as e:
        logger.error(f"Error sending message to Slack: {e.response.error}")
    except Exception as e:
        logger.error(f"Unexpected error sending message: {e}")

# Log user and bot interactions
def log_conversation(user_id, user_input, bot_response):
    log_entry = {
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "user_input": user_input,
        "bot_response": bot_response
    }
    logs_collection.insert_one(log_entry)

# Slack event handler
@app.post('/slack/events')
async def slack_events(req: Request):
    data = await req.json()
    logger.info(f"Received event: {data}")

    if data.get("type") == "url_verification":
        challenge = data.get("challenge")
        return JSONResponse(status_code=200, content={"challenge": challenge})

    if 'event' in data:
        event = data['event']

        # Handle file share events
        if 'files' in event:
            logger.info("Files found in the event")
            for file in event.get('files'):
                logger.info(f"File received: {file}")
                if file.get('filetype') == 'm4a':  # Corrected to check each file's type
                    logger.info("Audio m4a received")
                    file_url = file.get('url_private')
                    token = os.getenv("SLACK_BOT_TOKEN")

                    transcribed_text = process_audio_file(file_url, token)
                    if transcribed_text:
                        bot_response = generate_response(transcribed_text)
                        # Convert bot response to audio
                        audio_file_path = text_to_speech(bot_response)
                        logger.info(f"Generated audio file path: {audio_file_path}")
                        send_response_to_slack(event.get('channel'), bot_response, audio_file_path)
                    else:
                        send_response_to_slack(event.get('channel'), "Sorry, I couldn't understand the audio.")

        # Handle text message events
        elif event.get('type') == 'message' and 'subtype' not in event:
            logger.info("Processing text message")
            user_input = event.get('text', '').strip()
            user_id = event.get('user', '')
            channel = event.get('channel', '')

            # Get the bot's user ID to prevent it from responding to itself
            auth_response = slack_client.auth_test()
            bot_user_id = auth_response['user_id']

            if user_input and user_id and channel and user_id != bot_user_id:
                bot_response = generate_response(user_input)
                log_conversation(user_id, user_input, bot_response)
                send_response_to_slack(channel, bot_response)

    return JSONResponse(status_code=200, content={"status": "success"})


# Health check endpoint
@app.get("/healthz")
async def healthz():
    return {"status": "healthy"}

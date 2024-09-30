#!/usr/bin/env python
import io
import os
import logging
from datetime import datetime

import ffmpeg
from pydub import AudioSegment
from moviepy.editor import AudioFileClip
import requests

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from pymongo import MongoClient

import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient


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
logging.basicConfig(level=logging.DEBUG)
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


def generate_hypothetical_response(user_input: str) -> str:
    """
    Generates a hypothetical response using Azure OpenAI GPT-3.5 for the HyDE technique.
    
    Parameters:
    - user_input: str - The input text from the user.
    
    Returns:
    - str: The generated hypothetical document based on the input.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates hypothetical responses to help with query understanding."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=300,
            n=1,
            stop=None,
            temperature=0.7
        )
        hypothetical_response = response.choices[0].message.content.strip()
        logger.info(f"Generated hypothetical response: {hypothetical_response}")
        return hypothetical_response
    except Exception as e:
        logger.error(f"Error generating hypothetical response from Azure OpenAI: {e}")
        return "Unable to generate a hypothetical response at the moment."

def generate_embeddings(text: str) -> list:
    """
    Generates embeddings from a text using OpenAI's embedding API.
    
    Parameters:
    - text: str - The text to embed.
    
    Returns:
    - list: A list of embeddings (vectors) representing the text.
    """
    try:
        embeddings = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        embedding_vector = embeddings['data'][0]['embedding']
        logger.info(f"Generated embeddings for text")
        return embedding_vector
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return None

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two vectors.
    
    Parameters:
    - a: np.ndarray - The first vector.
    - b: np.ndarray - The second vector.
    
    Returns:
    - float: The cosine similarity score.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_cosmosdb(embedding: list) -> dict:
    """
    Searches for the closest document in CosmosDB based on the embedding similarity.
    
    Parameters:
    - embedding: list - The embedding of the hypothetical document.
    
    Returns:
    - dict: The closest matching document from CosmosDB.
    """
    try:
        # Convert the embedding to a numpy array
        query_embedding = np.array(embedding)

        # Fetch all documents and embeddings from CosmosDB (This assumes your documents have embeddings stored)
        documents = documents_collection.find()
        closest_document = None
        highest_similarity = -1

        # Iterate through each document and compute similarity
        for document in documents:
            document_embedding = np.array(document.get('embedding', []))  # Get stored embedding
            similarity = cosine_similarity(query_embedding, document_embedding)

            if similarity > highest_similarity:
                highest_similarity = similarity
                closest_document = document
        
        if closest_document:
            logger.info(f"Found the closest document with similarity: {highest_similarity}")
            return closest_document
        else:
            logger.info("No matching document found")
            return {"error": "No matching document found"}
    except Exception as e:
        logger.error(f"Error searching CosmosDB: {e}")
        return {"error": "An error occurred while searching for documents"}

def process_with_hyde(user_input: str) -> str:
    """
    Implements HyDE technique: generates a hypothetical document, embeds it, and searches CosmosDB.
    
    Parameters:
    - user_input: str - The input query from the user.
    
    Returns:
    - str: The response after applying the HyDE technique with CosmosDB search.
    """
    # Step 1: Generate a hypothetical document
    hypothetical_doc = generate_hypothetical_response(user_input)
    
    # Step 2: Generate embeddings for the hypothetical document
    embedding = generate_embeddings(hypothetical_doc)
    
    if embedding:
        # Step 3: Search CosmosDB for the most similar document
        closest_document = search_cosmosdb(embedding)
        if "error" in closest_document:
            return "Sorry, no relevant document found."
        else:
            return closest_document.get('content', 'Sorry, no relevant document found.')
    else:
        return "Unable to process the request using HyDE at the moment."

# --- End of HyDE Implementation ---

def generate_response(user_input: str) -> str:
    """
    Generates a chatbot response using the HyDE technique and CosmosDB search.
    
    Parameters:
    - user_input: str - The input text from the user.
    
    Returns:
    - str: The generated response or an error message if generation fails.
    """
    return process_with_hyde(user_input)

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

def convert_to_wav(input_file_path: str, output_file_path: str, file_type: str) -> str:
    """
    Converts an audio/video file to WAV format.
    
    Parameters:
    - input_file_path: str - The path to the input file.
    - output_file_path: str - The path to save the converted WAV file.
    - file_type: str - The type of file (e.g., 'm4a', 'mp4', 'webm').
    
    Returns:
    - str: The path to the converted WAV file.
    """
    try:
        if file_type == 'm4a':
            audio = AudioSegment.from_file(input_file_path, format="m4a")
            audio.export(output_file_path, format="wav")
        elif file_type in ['mp4', 'webm']:
            # Extract audio from video and convert to WAV
            with AudioFileClip(input_file_path) as audio_clip:
                audio_clip.write_audiofile(output_file_path, codec='pcm_s16le')
        else:
            logger.error(f"Unsupported file type: {file_type}")
            return None
        
        logger.info(f"Converted {input_file_path} to WAV: {output_file_path}")
        return output_file_path
    except Exception as e:
        logger.error(f"Failed to convert {input_file_path} to WAV: {str(e)}")
        return None

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

def process_audio_file(file_url: str, token: str) -> str:
    """
    Processes an audio file from Slack: downloads, uploads to Blob, converts to WAV, and transcribes in memory.
    
    Parameters:
    - file_url: str - The URL of the audio file to be processed.
    - token: str - The Slack authentication token for accessing the file.

    Returns:
    - str: The transcribed text or None if transcription fails.
    """
    headers = {
        "Authorization": f"Bearer {token}"
    }

    # Determine the file extension for processing
    file_extension = file_url.split('.')[-1]  # Get the file extension from the URL
    
    if file_extension not in ['m4a', 'mp4', 'webm']:
        logger.error(f"Unsupported file type: {file_extension}")
        return None

    try:
        # Download the file from Slack
        response = requests.get(file_url, headers=headers, stream=True)
        if response.status_code == 200:
            logger.info(f"File download from Slack successful. Status code: {response.status_code}")

            # Read the file content into memory (BytesIO)
            file_in_memory = io.BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                file_in_memory.write(chunk)
            file_in_memory.seek(0)  # Reset the buffer to the beginning

            # Convert the audio to WAV format in memory
            wav_in_memory = io.BytesIO()
            audio = AudioSegment.from_file(file_in_memory, format=file_extension)
            audio.export(wav_in_memory, format="wav")
            wav_in_memory.seek(0)

            # Upload the converted .wav file to Blob Storage from memory
            blob_name = "converted_audio.wav"
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(wav_in_memory, overwrite=True)
            logger.info(f".wav file uploaded to Blob Storage: {blob_client.url}")

            # Transcribe the .wav file using Azure Speech-to-Text
            transcribed_text = speech_to_text(blob_client.url)
            if transcribed_text:
                logger.info(f"Transcribed text: {transcribed_text}")
                return transcribed_text
            else:
                logger.error("Failed to transcribe audio using Azure Speech-to-Text")
                return None
        else:
            logger.error(f"Failed to download file from Slack. Status code: {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"Failed to process audio file: {str(e)}")
        return None

def generate_response(user_input: str) -> str:
    """
    Generates a chatbot response using HyDE first, and falls back to the original GPT-3.5 logic if HyDE fails.
    
    Parameters:
    - user_input: str - The input text from the user.
    
    Returns:
    - str: The generated response from HyDE or GPT-3.5, or an error message if both fail.
    """
    try:
        # Try the HyDE process first
        hyde_response = process_with_hyde(user_input)
        if hyde_response:
            logger.info(f"HyDE process successful, returning response: {hyde_response}")
            return hyde_response
        else:
            logger.warning("HyDE process did not return a valid result, falling back to GPT-3.5")

    except Exception as e:
        logger.error(f"Error during HyDE process: {e}")
        logger.warning("Falling back to GPT-3.5")

    # Fallback to original GPT-3.5 response generation
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
        logger.info(f"Generated bot response using GPT-3.5: {bot_response}")
        return bot_response

    except Exception as e:
        logger.error(f"Error generating response from GPT-3.5: {e}")
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
            logger.info(f"Uploading audio response to Slack: {file_path}")
            slack_client.files_upload(channels=channel, file=file_path, title="Response Audio")
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
            print("Processing text message")
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
        elif('files' in event.keys()):
            print("Files found in the event")
            logger.info("Files found in the event")
            logger.info(f"File received: {event['files'][0]['name']}")
            file_url = event['files'][0]['url_private']
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

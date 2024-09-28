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

# MongoDB connection for logging
mongo_client = MongoClient(os.getenv("COSMOS_DB_CONNECTION_STRING"))
db = mongo_client['soft_skills_chatbot']
logs_collection = db['logs']

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

# Log environment variables for debugging (only log that they exist, no values)
logger.info(f"AZURE_OPENAI_API_KEY exists: {bool(os.getenv('AZURE_OPENAI_API_KEY'))}")
logger.info(f"AZURE_OPENAI_ENDPOINT exists: {bool(os.getenv('AZURE_OPENAI_ENDPOINT'))}")

# Initialize the OpenAI client
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2023-03-15-preview"
)

# Initialize FastAPI app
app = FastAPI()

# Root route for health check or basic information
@app.get("/")
async def root():
    return {"message": "This is the Soft Skills Chatbot API."}

# Initialize Slack client with the bot token
slack_client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

# Azure Speech to Text
def speech_to_text(file_path):
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_KEY"), region=os.getenv("AZURE_REGION"))
    audio_config = speechsdk.audio.AudioConfig(filename=file_path)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    result = recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    else:
        return None

# Azure Text to Speech
def text_to_speech(response_text):
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_KEY"), region=os.getenv("AZURE_REGION"))
    audio_config = speechsdk.audio.AudioOutputConfig(filename="response_audio.wav")

    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    synthesizer.speak_text_async(response_text).get()
    return "response_audio.wav"

# Process audio files and convert to text
def process_audio_file(file_url, token):
    headers = {
        "Authorization": f"Bearer {token}"
    }
    response = requests.get(file_url, headers=headers)
    file_path = "received_audio.webm"

    with open(file_path, 'wb') as f:
        f.write(response.content)

    # Convert audio to text using Azure Speech-to-Text
    transcribed_text = speech_to_text(file_path)
    return transcribed_text

# Generate a response using Azure OpenAI's GPT model
def generate_response(user_input):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",  # Ensure this matches the deployment name in Azure
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
        logger.info(f"OpenAI API Response: {response}")
        bot_response = response.choices[0].message.content.strip()
        return bot_response
    except Exception as e:
        logger.error(f"Error generating response from Azure OpenAI: {e}")
        return "I'm sorry, I'm having trouble generating a response right now."

# Send response back to Slack as audio
def send_response_audio_to_slack(channel, file_path):
    try:
        slack_client.files_upload(channels=channel, file=file_path, title="Response Audio")
    except SlackApiError as e:
        logger.error(f"Error sending audio to Slack: {e.response.error}")
    except Exception as e:
        logger.error(f"Unexpected error sending audio to Slack: {e}")

# Send response back to Slack as text (for text inputs)
def send_response_to_slack(channel, response):
    try:
        slack_client.chat_postMessage(channel=channel, text=response)
    except SlackApiError as e:
        logger.error(f"Error sending message to Slack: {e.response.error}")
    except Exception as e:
        logger.error(f"Unexpected error sending message to Slack: {e}")

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

        # Handle file share events (audio)
        if event.get('subtype') == 'file_share' and event.get('files'):
            for file in event.get('files'):
                file_url = file.get('url_private')
                token = os.getenv("SLACK_BOT_TOKEN")
                transcribed_text = process_audio_file(file_url, token)
                
                if transcribed_text:
                    bot_response = generate_response(transcribed_text)
                    # Convert bot response to audio
                    audio_file_path = text_to_speech(bot_response)
                    # Send audio response back to Slack
                    send_response_audio_to_slack(event.get('channel'), audio_file_path)
                else:
                    send_response_to_slack(event.get('channel'), "Sorry, I couldn't understand the audio.")

        # Handle text message events
        elif event.get('type') == 'message' and 'subtype' not in event:
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

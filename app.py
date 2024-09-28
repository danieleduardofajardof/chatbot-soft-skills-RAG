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
from openai import AzureOpenAI

# Initialize Azure OpenAI client
openai_client = AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                            api_version="2023-03-15-preview")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Initialize Slack client with the bot token
slack_client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

# MongoDB connection
mongo_client = MongoClient(os.getenv("COSMOS_DB_CONNECTION_STRING"))
db = mongo_client['soft_skills_chatbot']
conversations_collection = db['conversations']
logs_collection = db['logs']

# Generate a response using Azure OpenAI's GPT model
def generate_response(user_input):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-35-turbo",
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
        return bot_response
    except Exception as e:
        logger.error(f"Error generating response from Azure OpenAI: {e}")
        return "I'm sorry, I'm having trouble generating a response right now."

# Send response back to Slack
def send_response_to_slack(channel, response):
    try:
        slack_client.chat_postMessage(channel=channel, text=response)
    except SlackApiError as e:
        logger.error(f"Error sending message: {e.response.error}")
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

# Save the conversation in MongoDB
def save_conversation(user_id, session_id, user_input, bot_response):
    conversation_entry = {
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "user_input": user_input,
        "bot_response": bot_response
    }
    conversations_collection.insert_one(conversation_entry)

# Azure Voice to Text (Optional)
def voice_to_text_azure():
    speech_key = os.getenv("AZURE_OPENAI_API_KEY")
    service_region = os.getenv("AZURE_REGION")

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Please speak something...")
    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"You said: {result.text}")
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized.")
        return ""
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Speech recognition canceled: {cancellation_details.reason}")
        return ""

# Azure Text to Voice (Optional)
def text_to_voice_azure(text):
    speech_key = os.getenv("AZURE_OPENAI_API_KEY")
    service_region = os.getenv("AZURE_REGION")

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    synthesizer.speak_text_async(text).get()

# Slack event handler
@app.post('/slack/events')
async def slack_events(req: Request):
    data = await req.json()
    logger.info(f"Received event: {data}")

    # Handle URL verification
    if data.get("type") == "url_verification":
        challenge = data.get("challenge")
        return JSONResponse(status_code=200, content={"challenge": challenge})

    # Handle event callbacks
    if 'event' in data:
        event = data['event']

        # Check if the event is a message from a user and not from the bot itself
        if event.get('type') == 'message' and 'subtype' not in event:
            user_input = event.get('text', '').strip()
            user_id = event.get('user', '')
            channel = event.get('channel', '')
            session_id = event.get('team', '')

            # Optional: Get the bot's user ID to prevent it from responding to itself
            try:
                auth_response = slack_client.auth_test()
                bot_user_id = auth_response['user_id']
            except SlackApiError as e:
                logger.error(f"Error fetching bot user ID: {e.response.error}")
                return JSONResponse(status_code=500, content={"status": "error"})

            if user_input and user_id and channel:
                if user_id == bot_user_id:
                    logger.info("Received a message from the bot itself. Ignoring.")
                    return JSONResponse(status_code=200, content={"status": "ignored"})

                # Generate the bot's response based on user input only
                bot_response = generate_response(user_input)

                # Log and save the conversation
                log_conversation(user_id, user_input, bot_response)
                save_conversation(user_id, session_id, user_input, bot_response)

                # Send response to Slack and convert it to speech (if applicable)
                send_response_to_slack(channel, bot_response)
                # Uncomment the following line if voice features are relevant
                # text_to_voice_azure(bot_response)  # Output the bot's response as speech

    return JSONResponse(status_code=200, content={"status": "success"})

# Health check endpoint
@app.get("/healthz")
async def healthz():
    return {"status": "healthy"}

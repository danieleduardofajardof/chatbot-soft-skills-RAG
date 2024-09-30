import os
import io
import logging
import requests
from azure.storage.blob import BlobServiceClient
from slack_sdk import WebClient
from datetime import datetime
from openai import AzureOpenAI
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from helper import generate_response, process_audio_file, send_response_to_slack, log_conversation

slack_client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
# Initialize FastAPI app
app = FastAPI()

# In-memory dictionary to store user context (multi-turn conversation state)
conversation_state = {}

# Logger
logger = logging.getLogger(__name__)

# Root health check endpoint
@app.get("/")
async def root() -> dict:
    return {"message": "This is the Soft Skills Chatbot API."}

# Health check endpoint
@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "healthy"}

# Slack event handler
@app.post('/slack/events')
async def slack_events(req: Request) -> JSONResponse:
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
            file_url = event['files'][0]['url_private']
            token = os.getenv("SLACK_BOT_TOKEN")
            transcribed_text = process_audio_file(file_url, token)
            user_id = event.get('user', '')
            channel = event.get('channel', '')
            
            if transcribed_text:
                bot_response = generate_response(user_id, transcribed_text, conversation_state)
                send_response_to_slack(channel, bot_response)
                log_conversation(user_id, transcribed_text, bot_response)
            else:
                send_response_to_slack(channel, "Sorry, I couldn't understand the audio.")

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
                # Generate a response based on conversation context
                bot_response = generate_response(user_id, user_input, conversation_state)
                log_conversation(user_id, user_input, bot_response)
                send_response_to_slack(channel, bot_response)

    return JSONResponse(status_code=200, content={"status": "success"})

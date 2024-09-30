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

# Configure the root logger
logging.basicConfig(
    level=logging.DEBUG,  # Set the root level to DEBUG for detailed logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler("app.log")  # File output
    ]
)

# Create a logger for your application
logger = logging.getLogger(__name__)

# Set logging level for specific libraries
logging.getLogger("requests").setLevel(logging.DEBUG)  # Requests library
logging.getLogger("slack_sdk").setLevel(logging.DEBUG)  # Slack SDK
logging.getLogger("azure.storage.blob").setLevel(logging.DEBUG)  # Azure Blob SDK
logging.getLogger("aiohttp").setLevel(logging.DEBUG)  # aiohttp library
logging.getLogger("pydub").setLevel(logging.INFO)  # Pydub (set to INFO for less verbosity)
logging.getLogger("openai").setLevel(logging.DEBUG)  # OpenAI Azure client

# Example usage: logging an event in your application
logger.info("Application started.")

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
            transcribed_text = await process_audio_file(file_url, token)  # Ensure async processing if required
            user_id = event.get('user', '')
            channel = event.get('channel', '')
            
            if transcribed_text:
                bot_response = await generate_response(user_id, transcribed_text, conversation_state)  # Ensure async if needed
                await send_response_to_slack(channel, bot_response)  # Ensure async if needed
                await log_conversation(user_id, transcribed_text, bot_response)  # Ensure async if needed
            else:
                await send_response_to_slack(channel, "Sorry, I couldn't understand the audio.")  # Ensure async

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
                bot_response = await generate_response(user_id, user_input, conversation_state)  # Ensure async
                await log_conversation(user_id, user_input, bot_response)  # Ensure async
                await send_response_to_slack(channel, bot_response)  # Ensure async

    return JSONResponse(status_code=200, content={"status": "success"})

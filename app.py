import logging
import os
from slack_sdk.web.async_client import AsyncWebClient
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from helper import process_audio_file, send_response_to_slack, log_conversation, generate_response, get_file_info_from_slack

slack_client = AsyncWebClient(token=os.getenv("SLACK_BOT_TOKEN"))
app = FastAPI()
conversation_state = {}

# Configure the root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/app.log")
    ]
)

logger = logging.getLogger(__name__)
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
            file = event['files'][0]
            file_url = file['url_private']
            token = os.getenv("SLACK_BOT_TOKEN")

            # Log the entire file object for inspection
            logger.info(f"File object from event: {file}")

            # Check if Slack has provided a transcription
            if 'transcription' in file and file['transcription'].get('status') == 'complete':
                transcribed_text = file['transcription'].get('text')
                logger.info(f"Using Slack transcription: {transcribed_text}")
            else:
                # Retry mechanism for Slack transcription (if it's still processing)
                logger.info("No transcription available or incomplete, retrying...")
                for attempt in range(5):  # Retry up to 5 times
                    await asyncio.sleep(5)  # Wait 5 seconds between retries
                    updated_file_info = await get_file_info_from_slack(file['id'], token)
                    if 'transcription' in updated_file_info and updated_file_info['transcription'].get('status') == 'complete':
                        transcribed_text = updated_file_info['transcription'].get('text')
                        logger.info(f"Slack transcription available after retry: {transcribed_text}")
                        break
                else:
                    logger.info("No transcription available after retries, processing audio manually")
                    transcribed_text = await process_audio_file(file_url, token)

            user_id = event.get('user', '')
            channel = event.get('channel', '')
            
            if transcribed_text:
                bot_response = generate_response(user_id, transcribed_text, conversation_state)
                await send_response_to_slack(channel, bot_response)
                log_conversation(user_id, transcribed_text, bot_response)
            else:
                await send_response_to_slack(channel, "Sorry, I couldn't understand the audio.")

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
                await send_response_to_slack(channel, bot_response)

    return JSONResponse(status_code=200, content={"status": "success"})

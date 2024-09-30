import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from helper import process_audio_file, send_response_to_slack, log_conversation, generate_response

app = FastAPI()
conversation_state = {}

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
            file_url = event['files'][0]['url_private']
            token = os.getenv("SLACK_BOT_TOKEN")
            transcribed_text = await process_audio_file(file_url, token)
            user_id = event.get('user', '')
            channel = event.get('channel', '')
            
            if transcribed_text:
                bot_response = await generate_response(user_id, transcribed_text, conversation_state)
                await send_response_to_slack(channel, bot_response)
                await log_conversation(user_id, transcribed_text, bot_response)
            else:
                await send_response_to_slack(channel, "Sorry, I couldn't understand the audio.")

        # Handle text message events
        elif event.get('type') == 'message' and 'subtype' not in event:
            logger.info("Processing text message")
            user_input = event.get('text', '').strip()
            user_id = event.get('user', '')
            channel = event.get('channel', '')

            if user_input and user_id and channel:
                bot_response = await generate_response(user_id, user_input, conversation_state)
                await log_conversation(user_id, user_input, bot_response)
                await send_response_to_slack(channel, bot_response)

    return JSONResponse(status_code=200, content={"status": "success"})

import io
import os
import logging
import requests
from pydub import AudioSegment
from azure.storage.blob import BlobServiceClient
from slack_sdk import WebClient
from datetime import datetime

# Initialize clients
slack_client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
container_client = blob_service_client.get_container_client(os.getenv("AZURE_STORAGE_CONTAINER_NAME"))

# Logger
logger = logging.getLogger(__name__)

# Generate response with basic multi-turn conversation handling
def generate_response(user_id: str, user_input: str, conversation_state: dict) -> str:
    """
    Generates a response and tracks conversation context for multi-turn conversations.
    """
    # Check if the user has an ongoing conversation
    last_question = conversation_state.get(user_id, {}).get("last_question")

    if last_question == "ask_for_name":
        bot_response = f"Nice to meet you, {user_input}!"
        conversation_state[user_id] = {"last_question": None}  # Clear context
    else:
        # First interaction or new context
        bot_response = "Hello! What is your name?"
        conversation_state[user_id] = {"last_question": "ask_for_name"}

    return bot_response

# Process the audio file in memory
def process_audio_file(file_url: str, token: str) -> str:
    headers = {"Authorization": f"Bearer {token}"}
    file_extension = file_url.split('.')[-1]
    if file_extension not in ['m4a', 'mp4', 'webm']:
        logger.error(f"Unsupported file type: {file_extension}")
        return None

    try:
        response = requests.get(file_url, headers=headers, stream=True)
        if response.status_code == 200:
            file_in_memory = io.BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                file_in_memory.write(chunk)
            file_in_memory.seek(0)

            wav_in_memory = io.BytesIO()
            audio = AudioSegment.from_file(file_in_memory, format=file_extension)
            audio.export(wav_in_memory, format="wav")
            wav_in_memory.seek(0)

            blob_name = "converted_audio.wav"
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(wav_in_memory, overwrite=True)

            logger.info(f".wav file uploaded to Blob Storage: {blob_client.url}")

            # Transcribe the .wav file (use an actual transcription service like Azure)
            transcribed_text = "Dummy transcription text"
            return transcribed_text
        else:
            logger.error(f"Failed to download file. Status code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
        return None

# Send a response back to Slack
def send_response_to_slack(channel: str, response: str, file_path: str = None) -> None:
    try:
        if file_path:
            slack_client.files_upload(channels=channel, file=file_path, title="Response Audio")
        else:
            slack_client.chat_postMessage(channel=channel, text=response)
    except Exception as e:
        logger.error(f"Error sending message to Slack: {e}")

# Log the conversation in a database
def log_conversation(user_id: str, user_input: str, bot_response: str) -> None:
    log_entry = {
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "user_input": user_input,
        "bot_response": bot_response
    }
    # Here you'd insert the log into MongoDB or another storage system.
    logger.info(f"Conversation logged: {log_entry}")

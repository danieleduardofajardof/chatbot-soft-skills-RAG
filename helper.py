import os
import io
import logging
import requests
from azure.storage.blob import BlobServiceClient
from slack_sdk import WebClient
from datetime import datetime
from openai import AzureOpenAI
import aiohttp
from pydub import AudioSegment

# Initialize clients
slack_client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
container_client = blob_service_client.get_container_client(os.getenv("AZURE_STORAGE_CONTAINER_NAME"))
openai_client = AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"), azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))

# Configure the root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(), 
        logging.FileHandler("app.log")
    ]
)

logger = logging.getLogger(__name__)

# Set logging level for specific libraries
logging.getLogger("requests").setLevel(logging.DEBUG)
logging.getLogger("slack_sdk").setLevel(logging.DEBUG)
logging.getLogger("azure.storage.blob").setLevel(logging.DEBUG)
logging.getLogger("aiohttp").setLevel(logging.DEBUG)
logging.getLogger("pydub").setLevel(logging.INFO)
logging.getLogger("openai").setLevel(logging.DEBUG)

logger.info("Application started.")

def analyze_sentiment_gpt(user_input: str) -> str:
    try:
        prompt = f"Please classify the sentiment of the following text as 'positive', 'neutral', or 'negative':\n\n{user_input}"
        response = openai_client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that analyzes sentiment."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0
        )
        sentiment = response.choices[0].message.content.strip().lower()
        logger.info(f"GPT-3.5 sentiment analysis result: {sentiment}")
        return sentiment if sentiment in ["positive", "neutral", "negative"] else "neutral"

    except Exception as e:
        logger.error(f"Error during GPT-3.5 sentiment analysis: {e}")
        return "neutral"

def generate_response(user_id: str, user_input: str, conversation_state: dict) -> str:
    sentiment = analyze_sentiment_gpt(user_input)
    last_question = conversation_state.get(user_id, {}).get("last_question")

    if last_question == "ask_for_name":
        bot_response = f"Nice to meet you, {user_input}." if sentiment == "neutral" else f"Great to meet you, {user_input}!" if sentiment == "positive" else f"I'm sorry if you're feeling down, {user_input}. I'm here if you want to talk!"
        conversation_state[user_id] = {"last_question": None}
    else:
        bot_response = "Hello! What is your name?" if sentiment == "neutral" else "Hello! It's great to hear from you! What can I help you with today?" if sentiment == "positive" else "It sounds like something might be bothering you. Would you like to share more?"
        conversation_state[user_id] = {"last_question": "ask_for_name"}

    return bot_response
import os
import io
import logging
import requests
from azure.storage.blob import BlobServiceClient
from slack_sdk import WebClient
from datetime import datetime
from openai import AzureOpenAI
import aiohttp
from pydub import AudioSegment

# Initialize clients
slack_client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
container_client = blob_service_client.get_container_client(os.getenv("AZURE_STORAGE_CONTAINER_NAME"))
openai_client = AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"), azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))

# Configure the root logger
logging.basicConfig(
    level=logging.DEBUG,  # Set the root level to DEBUG for detailed logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler("app.log")  # File output
    ]
)

logger = logging.getLogger(__name__)

# Ensure the container exists
try:
    container_client.create_container()
except Exception as e:
    logger.info(f"Container already exists or unable to create: {str(e)}")

# Set logging level for specific libraries
logging.getLogger("requests").setLevel(logging.DEBUG)
logging.getLogger("slack_sdk").setLevel(logging.DEBUG)
logging.getLogger("azure.storage.blob").setLevel(logging.DEBUG)
logging.getLogger("aiohttp").setLevel(logging.DEBUG)
logging.getLogger("pydub").setLevel(logging.INFO)
logging.getLogger("openai").setLevel(logging.DEBUG)

logger.info("Application started.")

async def process_audio_file(file_url: str, token: str) -> str:
    """
    Downloads, converts to WAV, uploads to Blob Storage asynchronously, and returns a transcription.
    """
    headers = {"Authorization": f"Bearer {token}"}
    file_extension = file_url.split('.')[-1]

    if file_extension not in ['m4a', 'mp4', 'webm']:
        logger.error(f"Unsupported file type: {file_extension}")
        return None

    try:
        logger.info(f"Starting to download file from URL: {file_url}")

        async with aiohttp.ClientSession() as session:
            async with session.get(file_url, headers=headers) as response:
                if response.status == 200:
                    logger.info(f"Successfully downloaded file from Slack. Status: {response.status}")
                    file_in_memory = io.BytesIO(await response.read())
                    file_in_memory.seek(0)

                    # Convert the audio file to WAV
                    wav_in_memory = io.BytesIO()
                    audio = AudioSegment.from_file(file_in_memory, format=file_extension)
                    audio.export(wav_in_memory, format="wav")
                    wav_in_memory.seek(0)
                    logger.info(f"Converted audio to WAV in memory.")

                    # Upload to Azure Blob Storage
                    blob_name = "converted_audio.wav"
                    blob_client = container_client.get_blob_client(blob_name)

                    try:
                        logger.info(f"Starting to upload .wav file to Blob Storage: {blob_name}")
                        await blob_client.upload_blob(wav_in_memory, overwrite=True)
                        logger.info(f".wav file uploaded successfully to Blob Storage. URL: {blob_client.url}")
                    except Exception as e:
                        logger.error(f"Error during Blob Storage upload: {e}")
                        return None

                    # If upload successful, return dummy transcription
                    return "Dummy transcription"
                else:
                    logger.error(f"Failed to download file from Slack. Status code: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
        return None

async def send_response_to_slack(channel: str, response: str, file_path: str = None) -> None:
    try:
        if file_path:
            logger.info(f"Sending file response to Slack: {file_path}")
            await slack_client.files_upload(channels=channel, file=file_path, title="Response Audio")
        else:
            logger.info(f"Sending text response to Slack: {response}")
            await slack_client.chat_postMessage(channel=channel, text=response)
    except Exception as e:
        logger.error(f"Error sending message or file to Slack: {e}")


def log_conversation(user_id: str, user_input: str, bot_response: str) -> None:
    log_entry = {
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "user_input": user_input,
        "bot_response": bot_response
    }
    logger.info(f"Conversation logged: {log_entry}")

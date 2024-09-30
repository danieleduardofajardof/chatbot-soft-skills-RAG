import os
import io
import logging
import aiohttp
from slack_sdk.web.async_client import AsyncWebClient
from azure.storage.blob.aio import BlobServiceClient
from azure.cognitiveservices.speech import SpeechConfig, AudioConfig
from azure.cognitiveservices.speech.aio import SpeechRecognizer
from pydub import AudioSegment
from datetime import datetime

# Initialize clients
slack_client = AsyncWebClient(token=os.getenv("SLACK_BOT_TOKEN"))
blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
container_client = blob_service_client.get_container_client(os.getenv("AZURE_STORAGE_CONTAINER_NAME"))
speech_config = SpeechConfig(subscription=os.getenv("AZURE_SPEECH_API_KEY"), region=os.getenv("AZURE_REGION"))

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

async def process_audio_file(file_url: str, token: str) -> str:
    """
    Downloads, converts to WAV, and transcribes audio file asynchronously using Azure Speech-to-Text.
    """
    headers = {"Authorization": f"Bearer {token}"}
    file_extension = file_url.split('.')[-1]

    if file_extension not in ['m4a', 'mp4', 'webm']:
        logger.error(f"Unsupported file type: {file_extension}")
        return None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url, headers=headers) as response:
                if response.status == 200:
                    file_in_memory = io.BytesIO(await response.read())
                    file_in_memory.seek(0)

                    # Convert audio to WAV in memory
                    wav_in_memory = io.BytesIO()
                    audio = AudioSegment.from_file(file_in_memory, format=file_extension)
                    audio.export(wav_in_memory, format="wav")
                    wav_in_memory.seek(0)

                    # Transcribe the audio using Azure Speech-to-Text asynchronously
                    audio_config = AudioConfig(stream=io.BytesIO(wav_in_memory.getvalue()))
                    recognizer = SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
                    result = await recognizer.recognize_once_async()

                    if result.reason == result.Reason.RecognizedSpeech:
                        logger.info(f"Transcription result: {result.text}")
                        return result.text
                    else:
                        logger.error("No speech could be recognized")
                        return None
                else:
                    logger.error(f"Failed to download file from Slack. Status code: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
        return None

async def send_response_to_slack(channel: str, response: str) -> None:
    """
    Sends the bot's response back to Slack asynchronously.
    """
    try:
        logger.info(f"Sending response to Slack: {response}")
        await slack_client.chat_postMessage(channel=channel, text=response)
    except Exception as e:
        logger.error(f"Error sending message to Slack: {e}")

async def log_conversation(user_id: str, user_input: str, bot_response: str) -> None:
    """
    Logs conversation details asynchronously.
    """
    log_entry = {
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "user_input": user_input,
        "bot_response": bot_response
    }
    logger.info(f"Conversation logged: {log_entry}")

async def upload_to_blob(wav_in_memory: io.BytesIO, blob_name: str) -> str:
    """
    Uploads the in-memory WAV file to Azure Blob Storage asynchronously.
    """
    try:
        blob_client = container_client.get_blob_client(blob_name)
        await blob_client.upload_blob(wav_in_memory, overwrite=True)
        logger.info(f".wav file uploaded to Blob Storage: {blob_client.url}")
        return blob_client.url
    except Exception as e:
        logger.error(f"Error uploading file to Blob Storage: {e}")
        return None

async def analyze_sentiment_gpt(user_input: str) -> str:
    """
    Analyzes sentiment using GPT-3.5 by asking it to classify the sentiment.

    Parameters:
    - user_input: str - The input text from the user.

    Returns:
    - str: "positive", "neutral", or "negative" sentiment classification.
    """
    try:
        prompt = f"Please classify the sentiment of the following text as 'positive', 'neutral', or 'negative':\n\n{user_input}"
        response = await openai_client.chat.completions.create(
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

        if sentiment in ["positive", "neutral", "negative"]:
            return sentiment
        else:
            return "neutral"  # Default to neutral if GPT-3.5 gives an unexpected response

    except Exception as e:
        logger.error(f"Error during GPT-3.5 sentiment analysis: {e}")
        return "neutral"  # Default to neutral if there's an error

async def generate_response(user_id: str, user_input: str, conversation_state: dict) -> str:
    """
    Generates a response with sentiment analysis using GPT-3.5.
    """
    # Analyze sentiment using GPT-3.5
    sentiment = await analyze_sentiment_gpt(user_input)
    
    # Check if the user has an ongoing conversation
    last_question = conversation_state.get(user_id, {}).get("last_question")

    # Adjust response based on sentiment and context
    if last_question == "ask_for_name":
        if sentiment == "negative":
            bot_response = f"I'm sorry if you're feeling down, {user_input}. I'm here if you want to talk!"
        elif sentiment == "positive":
            bot_response = f"Great to meet you, {user_input}!"
        else:
            bot_response = f"Nice to meet you, {user_input}."
        conversation_state[user_id] = {"last_question": None}  # Clear context
    else:
        if sentiment == "negative":
            bot_response = "Hello! It sounds like something might be bothering you. Would you like to share more?"
        elif sentiment == "positive":
            bot_response = "Hello! It's great to hear from you! What can I help you with today?"
        else:
            bot_response = "Hello! What is your name?"
        conversation_state[user_id] = {"last_question": "ask_for_name"}

    return bot_response

async def get_file_info_from_slack(file_id: str, token: str) -> dict:
    url = f"https://slack.com/api/files.info?file={file_id}"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                file_info = await response.json()
                logger.info(f"Retrieved file info: {file_info}")
                return file_info.get('file', {})
            else:
                logger.error(f"Failed to retrieve file info from Slack. Status code: {response.status}")
                return {}

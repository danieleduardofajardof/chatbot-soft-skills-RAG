import os
import io
import logging
import requests
from azure.storage.blob import BlobServiceClient
from slack_sdk import WebClient
from datetime import datetime
from openai import AzureOpenAI

# Initialize clients
slack_client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
container_client = blob_service_client.get_container_client(os.getenv("AZURE_STORAGE_CONTAINER_NAME"))
openai_client = AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"), azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))

# Logger
logger = logging.getLogger(__name__)

def analyze_sentiment_gpt(user_input: str) -> str:
    """
    Analyzes sentiment using GPT-3.5 by asking it to classify the sentiment.

    Parameters:
    - user_input: str - The input text from the user.

    Returns:
    - str: "positive", "neutral", or "negative" sentiment classification.
    """
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

        if sentiment in ["positive", "neutral", "negative"]:
            return sentiment
        else:
            return "neutral"  # Default to neutral if GPT-3.5 gives an unexpected response

    except Exception as e:
        logger.error(f"Error during GPT-3.5 sentiment analysis: {e}")
        return "neutral"  # Default to neutral if there's an error

def generate_response(user_id: str, user_input: str, conversation_state: dict) -> str:
    """
    Generates a response with sentiment analysis using GPT-3.5.
    """
    # Analyze sentiment using GPT-3.5
    sentiment = analyze_sentiment_gpt(user_input)
    
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
            return "Dummy transcription"  # You can replace this with an actual transcription service
        else:
            logger.error(f"Failed to download file from Slack. Status code: {response.status_code}")
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

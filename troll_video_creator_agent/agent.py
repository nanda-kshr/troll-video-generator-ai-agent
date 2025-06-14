import base64
import datetime
import json
from zoneinfo import ZoneInfo
from google.adk.agents import Agent, LlmAgent
import ffmpeg
from google.cloud import speech
import os

import shutil
from google.adk.artifacts import InMemoryArtifactService
from google.adk.tools import ToolContext
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import FunctionTool
from google.genai import types
from typing import Optional, Dict, Any
import logging
from google.adk.events import Event, EventActions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def extract_audio(video_path, audio_path):
    """Extract audio from a video file using FFmpeg."""
    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, audio_path, acodec='mp3', audio_bitrate='160k')
        ffmpeg.run(stream)
        print(f"Audio extracted to {audio_path}")
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e.stderr.decode()}")
        raise

def transcribe_audio(audio_path, language_code="en-US"):
    """Transcribe audio with timestamps using Google Cloud Speech-to-Text."""
    client = speech.SpeechClient()

    # If audio is short (<60s), read directly; otherwise, upload to Google Cloud Storage
    if os.path.getsize(audio_path) < 10_000_000:  # Less than 10MB
        with open(audio_path, "rb") as audio_file:
            content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)
    else:
        # Upload to Google Cloud Storage (requires gsutil or google-cloud-storage)
        # Example: gs://your-bucket/audio.mp3
        print("Audio too large. Upload to Google Cloud Storage and provide URI.")
        return

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=16000,  # Adjust if needed (check with ffprobe)
        language_code=language_code,
        enable_word_time_offsets=True,  # Enable timestamps
    )

    # Perform transcription
    operation = client.long_running_recognize(config=config, audio=audio)
    print("Transcribing audio...")
    response = operation.result(timeout=600)  # Timeout after 10 minutes
    print("Transcribed")

    # Process results
    transcript_data = []
    for result in response.results:
        alternative = result.alternatives[0]
        transcript = alternative.transcript
        words = [
            {
                "word": word_info.word,
                "start_time": word_info.start_time.total_seconds(),
                "end_time": word_info.end_time.total_seconds(),
            }
            for word_info in alternative.words
        ]
        transcript_data.append({"transcript": transcript, "words": words})

    return transcript_data


async def create_troll(filename: str, tool_context: ToolContext) -> None:
    """
    Process video input from ADK web UI.
    
    Args:
        filename: The filename of the artifact.
        document_bytes: Bytes of the uploaded file.
        tool_context: The context object provided by the ADK framework
        
    Returns:
        A dictionary containing:
        - status: A string indicating the status of video processing
        - video: The processed video part (if successful)
    """

    try:
        logger.info("Received video input from ADK web UI")
        audio_file_name= "audio_"+filename+".mp3"
        extract_audio(filename, audio_file_name)
        transcript_data = transcribe_audio(audio_file_name)
        logger.info(f"Transcription data for {transcript_data}:")
        result=[]
        for entry in transcript_data:
            transcript = entry['transcript']
            words = entry['words']
            if words:
                start_time = words[0]['start_time'] 
                end_time = words[-1]['end_time']    
                result.append({
                    'entire_sentence_transcript': transcript.strip(),
                    'start': start_time,
                    'end': end_time
            })
        
        artifact_id = f"modified_transcript_{datetime.datetime.now(ZoneInfo('UTC')).isoformat()}.json"
        tool_context.artifact_service.put(
            artifact_id=artifact_id,
            content=json.dumps(result).encode('utf-8'),
            content_type='application/json'
        )
        logger.info(f"Modified transcript saved as artifact: {artifact_id}")
        
        return {
            "status": "Transcript modified successfully",
            "artifact_id": artifact_id,
            "modified_data": result
        }
    except Exception as e:
        logger.error(f"Error modifying transcript: {str(e)}")
        return {"status": f"Error modifying transcript: {str(e)}"}
    
# Initialize services
artifact_service = InMemoryArtifactService()
session_service = InMemorySessionService()

# Create the agent with video processing capabilities
root_agent = LlmAgent(
    name="troll_generator",
    model="gemini-2.0-flash",
    description=(
        "This agent creates troll videos based on user queries. "
        "It accepts video input from the ADK web UI and processes it to create entertaining content."
    ),
    instruction=(
        "You are a video creation agent that can create troll videos based on user queries. "
        "You can accept video name input from the ADK web UI and process it to create entertaining content. "
        "When a user uploads a video, it will be available as a video part in the input."
        "After creating video, you should return both the status message and the processed video back to the user."
    ),
    tools=[
        FunctionTool(create_troll)
    ]
)
#my_agent = (name="artifact_user_agent", model="gemini-2.0-flash")

# Initialize the runner
runner = Runner(
    agent=root_agent,
    app_name="video_agent_app",
    session_service=session_service,
    artifact_service=artifact_service
)


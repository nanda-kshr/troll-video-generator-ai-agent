import base64
import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.cloud import videointelligence
import os
from google.adk.artifacts import InMemoryArtifactService
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.adk.runners import Runner
from google.adk.tools import FunctionTool
from google.genai import types
from typing import Optional
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_troll_video(filename: str, tool_context: ToolContext) -> Optional[str]:
    """
    Loads a previously saved image artifact and returns its description (placeholder).

    Args:
        filename: The filename of the artifact to load.
        tool_context: The context object provided by the ADK framework.

    Returns:
        A description of the image, or None if not found.
    """

    print(f"--- Tool: Attempting to load artifact '{filename}' ---")
    try:
        # Use the context to load the artifact (latest version)
        loaded_artifact: Optional[types.Part] = tool_context.load_artifact(filename=filename)

        if loaded_artifact:
            print(f"--- Tool: Loaded artifact '{filename}' (mime_type: {loaded_artifact.inline_data.mime_type}) ---")
            # --- In a real tool, you might send this image data to another model for analysis ---
            # --- or pass it back to the primary LLM ---
            # For simplicity, we just return a placeholder description
            return f"Loaded artifact '{filename}'. It appears to be an image of type {loaded_artifact.inline_data.mime_type}."
        else:
            print(f"--- Tool: Artifact '{filename}' not found. ---")
            return f"Artifact '{filename}' could not be found."
    except Exception as e:
        print(f"--- Tool: Error loading artifact: {e} ---")
        return f"Error loading artifact '{filename}': {e}"

    
    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.SPEECH_TRANSCRIPTION]
    config = videointelligence.SpeechTranscriptionConfig(
        language_code="en-US", enable_automatic_punctuation=True
    )
    video_context = videointelligence.VideoContext(speech_transcription_config=config)
    # Save the video file to current folder
    timestamp = datetime.datetime.now(ZoneInfo("UTC")).strftime("%Y%m%d_%H%M%S")
    filename = f"input_video_{timestamp}.mp4"
    path = os.path.join(os.getcwd(), filename)

    with open(path, 'wb') as f:
        f.write(file_data)
    operation = video_client.annotate_video(
        request={
            "features": features,
            "input_uri": path,
            "video_context": video_context,
        }
    )

    print("\nProcessing video for speech transcription.")

    result = operation.result(timeout=600)
    annotation_results = result.annotation_results[0]
    for speech_transcription in annotation_results.speech_transcriptions:
        # The number of alternatives for each transcription is limited by
        # SpeechTranscriptionConfig.max_alternatives.
        # Each alternative is a different possible transcription
        # and has its own confidence score.
        for alternative in speech_transcription.alternatives:
            print("Alternative level information:")

            print("Transcript: {}".format(alternative.transcript))
            print("Confidence: {}\n".format(alternative.confidence))

            print("Word level information:")
            for word_info in alternative.words:
                word = word_info.word
                start_time = word_info.start_time
                end_time = word_info.end_time
                print(
                    "\t{}s - {}s: {}".format(
                        start_time.seconds + start_time.microseconds * 1e-6,
                        end_time.seconds + end_time.microseconds * 1e-6,
                        word,
                    )
                )

    


artifact_service = InMemoryArtifactService()  # Use GcsArtifactService for production
session_service = InMemorySessionService()

root_agent = Agent(
    name="troll_generator",
    model="gemini-2.0-flash",
    description=(
        "This agent creates troll videos based on user queries. "
    ),
    instruction=(
        "You are a video creation agent that can create troll videos based on user queries. "
    ),
    tools=[FunctionTool(func=create_troll_video, name="create_troll_video")]
)

runner = Runner(
    agent=root_agent,
    app_name="video_agent_app",
    session_service=session_service,
    artifact_service=artifact_service
)
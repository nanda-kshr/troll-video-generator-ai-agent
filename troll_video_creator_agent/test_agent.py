import os
from agent import runner, process_video_input, create_troll_video
from google.adk.tools import ToolContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService, Session
from google.adk.agents import Agent
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_agent() -> Agent:
    """Create a mock agent for testing."""
    return Agent(
        name="test_agent",
        model="gemini-2.0-flash",
        description="Test agent for video processing",
        instruction="Process video input for testing"
    )

def test_video_processing():
    # Create a test video file path
    test_video_path = "its.mp4"
    
    try:
        # Create services
        artifact_service = InMemoryArtifactService()
        session_service = InMemorySessionService()
        
        # Create a test session
        session = Session(
            id=str(uuid.uuid4()),
            app_name="video_agent_app",
            user_id="test_user"
        )
        
        # Create invocation context
        invocation_context = InvocationContext(
            invocation_id=f"e-{str(uuid.uuid4())}",
            agent=create_mock_agent(),
            session=session,
            session_service=session_service,
            artifact_service=artifact_service
        )
        
        # Initialize tool context
        tool_context = ToolContext(
            invocation_context=invocation_context,
            function_call_id="test_function_call"
        )
        
        # Read the existing video file
        with open(test_video_path, "rb") as f:
            video_content = f.read()
        
        # Create video part
        video_part = types.Part(
            inline_data=types.Blob(
                mime_type="video/mp4",
                data=video_content
            )
        )
        
        # Simulate video data from ADK web UI
        video_data = {
            "video": video_part
        }
        
        # Test process_video_input
        logger.info("Testing process_video_input...")
        result = process_video_input(video_data, tool_context)
        logger.info(f"Process video result: {result}")
        
        # If video was saved successfully, test create_troll_video
        if "successfully" in result.lower():
            # Extract the artifact name from the result
            artifact_name = result.split("as ")[-1].strip()
            logger.info(f"Testing create_troll_video with artifact: {artifact_name}")
            troll_result = create_troll_video(artifact_name, tool_context)
            logger.info(f"Create troll video result: {troll_result}")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise  # Re-raise the exception to see the full traceback

if __name__ == "__main__":
    test_video_processing() 
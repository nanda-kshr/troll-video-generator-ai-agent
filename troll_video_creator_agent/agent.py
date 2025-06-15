import base64
import datetime
import json
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from google.adk.agents import Agent, LlmAgent
import ffmpeg
from google.cloud import speech
import os
import shutil
from google.adk.tools import ToolContext
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import FunctionTool
from google.genai import types
from typing import Optional, Dict, Any, List
import logging
from google import genai
from google.cloud import storage 
from .dataset import TROLL_CLIPS  

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GCS_BUCKET_NAME = "troll_video_gen" # <--- IMPORTANT: Replace with your actual GCS bucket name
GCS_UPLOAD_FOLDER = "troll_video_gen_audios" # Optional: folder within your bucket

def upload_to_gcs(source_file_name, destination_blob_name):
    """Uploads a file to the Google Cloud Storage bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name} in bucket {GCS_BUCKET_NAME}.")
    return f"gs://{GCS_BUCKET_NAME}/{destination_blob_name}"



# Troll clips dataset



def extract_audio(video_path, audio_path):
    """Extract audio from a video file using FFmpeg."""
    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, audio_path, acodec='mp3', audio_bitrate='160k', ar=44100)
        ffmpeg.run(stream)
        print(f"Audio extracted to {audio_path}")
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e.stderr.decode()}")
        raise

def transcribe_audio(audio_path, language_code="en-US"):
    """Transcribe audio with timestamps using Google Cloud Speech-to-Text."""
    client = speech.SpeechClient()

    # Upload to GCS
    gcs_destination_path = f"{GCS_UPLOAD_FOLDER}/{os.path.basename(audio_path)}"
    gcs_uri = upload_to_gcs(audio_path, gcs_destination_path)
    audio = speech.RecognitionAudio(uri=gcs_uri)
    logger.info(f"Using GCS URI for transcription: {gcs_uri}")

    # Probe audio to get sample rate
    probe = ffmpeg.probe(audio_path)
    audio_stream = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
    sample_rate = int(audio_stream.get('sample_rate', 44100))

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=sample_rate,  # Use probed sample rate
        language_code=language_code,
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
        audio_channel_count=2,
        enable_separate_recognition_per_channel=False,
        max_alternatives=1,
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    logger.info("Transcribing audio...")
    try:
        response = operation.result(timeout=3600)  # 1-hour timeout
        logger.info("Transcribed")
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise

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

    # Log transcription duration
    total_duration = max(word['end_time'] for entry in transcript_data for word in entry['words']) if transcript_data else 0
    logger.info(f"Transcription duration: {total_duration} seconds")
    return transcript_data

def modify_transcript(transcript_data: List[Dict], tool_context: ToolContext) -> Dict[str, Any]:
    """Prompt Gemini to add troll clips to the transcript and return updated data."""
    try:
        logger.info("Modifying transcript with troll clips")

        # Prepare prompt for Gemini
        prompt = (
            "You are a video editing assistant specializing in comedic troll video creation. Given a transcript with sentences and their start/end times, "
            "decide where to insert troll video clips from the provided dataset to maximize comedic effect. Follow these rules:\n"
            "0. Be a little rude and you are free to use bad words trolls from the dataset .\n"
            "1. Analyze the sentiment and context of each sentence (e.g., positive, negative, neutral, sarcastic, comedic).\n"
            "2. Assign a troll clip only when its description matches the sentence's sentiment or context.\n"
            "3. Do not assign trolls to neutral or very short sentences (less than 3 words) unless they have a clear comedic or mistaken context.\n"
            "4. Limit troll clips to at most one per every two sentences to avoid overuse.\n"
            "5. Return the transcript with an additional 'troll' field for each sentence, containing the troll clip filename (e.g., '3 .mp4') or None if no troll is added.\n\n"
            "Transcript:\n" + json.dumps(transcript_data, indent=2) + "\n\n"
            "Troll Clips Dataset:\n" + json.dumps(TROLL_CLIPS, indent=2) + "\n\n"
            "Example Output:\n"
            "[\n"
            "  {'entire_sentence_transcript': 'Hey how are you? you look so beautiful', 'start': 0, 'end': 2, 'troll': '3 .mp4'},\n"
            "  {'entire_sentence_transcript': 'ok bie', 'start': 2, 'end': 3, 'troll': None}\n"
            "]\n\n"
            "Provide the modified transcript array in JSON format.\n"
            "Remember to respond with all the sentences in the transcript, even if no troll is added."
        )
        load_dotenv()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(system_instruction=prompt),
            contents=json.dumps(transcript_data, indent=2)
        )
        modified_data = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
        logger.info(f"\n\n\nModified transcript data: {modified_data}\n\n\n")
        # Save modified data to local file
        modified_file = "modified_transcript.json"
        with open(modified_file, "w", encoding="utf-8") as f:
            json.dump(modified_data, f, indent=2)
        logger.info(f"Modified transcript saved to {modified_file}")

        return {
            "status": "Transcript modified with troll clips successfully",
            "modified_file": modified_file,
            "modified_data": modified_data
        }
    except Exception as e:
        logger.error(f"Error modifying transcript: {str(e)}")
        return {"status": f"Error modifying transcript: {str(e)}"}

def merge_video_with_trolls(video_path: str, timeline: List[Dict], output_path: str) -> None:
    """Cut input video and insert troll clips between segments based on the timeline."""
    try:
        logger.info("Merging video with troll clips")
        temp_file_list = "concat_list.txt"
        streams = []

        # Get input video properties for normalization
        probe = ffmpeg.probe(video_path)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        input_width = video_stream['width']
        input_height = video_stream['height']
        input_fps = eval(video_stream.get('r_frame_rate', '30/1'))  # Default to 30fps
        input_aspect = video_stream.get('display_aspect_ratio', f"{input_width}:{input_height}")
        sar = video_stream.get('sample_aspect_ratio', '1:1')  # Sample aspect ratio

        for i, segment in enumerate(timeline):
            start = segment["start"]
            duration = segment["end"] - segment["start"]

            # Handle input video segment
            segment_path = f"temp_segment_{i}.mp4"
            stream = ffmpeg.input(video_path, ss=start, t=duration)
            # Re-encode input video segment to ensure consistency
            stream = ffmpeg.output(
                stream,
                segment_path,
                vcodec='libx264',
                acodec='aac',
                video_bitrate='2000k',
                audio_bitrate='192k',
                r=input_fps,  # Match frame rate
                vf=f'scale={input_width}:{input_height}:force_original_aspect_ratio=decrease,pad={input_width}:{input_height}:(ow-iw)/2:(oh-ih)/2,setsar={sar}',  # Preserve aspect ratio
                format='mp4',
                movflags='faststart'
            )
            ffmpeg.run(stream, overwrite_output=True)
            streams.append(segment_path)

            # Handle troll clip if assigned
            troll = segment.get("troll")
            if troll and troll in TROLL_CLIPS:
                troll_path = os.path.join("troll_video_creator_agent/trollclips", troll)
                if os.path.exists(troll_path):
                    reencoded_troll = f"temp_troll_{i}.mp4"
                    # Re-encode troll clip to match input video properties
                    stream = ffmpeg.input(troll_path)
                    stream = ffmpeg.output(
                        stream,
                        reencoded_troll,
                        vcodec='libx264',
                        acodec='aac',
                        video_bitrate='2000k',
                        audio_bitrate='192k',
                        r=input_fps,
                        vf=f'scale={input_width}:{input_height}:force_original_aspect_ratio=decrease,pad={input_width}:{input_height}:(ow-iw)/2:(oh-ih)/2,setsar={sar}',
                        format='mp4',
                        movflags='faststart'
                    )
                    ffmpeg.run(stream, overwrite_output=True)
                    streams.append(reencoded_troll)
                else:
                    logger.warning(f"Troll clip {troll_path} not found, skipping")

        # Write file list for concat
        with open(temp_file_list, "w") as f:
            for stream in streams:
                f.write(f"file '{stream}'\n")

        # Concatenate all segments with dynamic stream mapping
        concat_stream = ffmpeg.input(temp_file_list, f='concat', safe=0)
        concat_stream = ffmpeg.output(
            concat_stream,
            "output/"+output_path,
            vcodec='libx264',
            acodec='aac',
            video_bitrate='2000k',
            audio_bitrate='192k',
            map='0',  # Map all streams (video and audio)
            format='mp4',
            movflags='faststart'
        )
        ffmpeg.run(concat_stream, overwrite_output=True)

        # Clean up temporary files
        os.remove(temp_file_list)
        for segment in [s for s in streams if s.startswith("temp_segment_") or s.startswith("temp_troll_")]:
            os.remove(segment)

        logger.info(f"Merged video saved as output/{output_path}")
    except Exception as e:
        logger.error(f"Error merging video: {str(e)}")
        raise


async def summarize_transcript(modified_file: str) -> Dict[str, Any]:
    """Summarize the modified transcript data from local file."""
    try:
        logger.info(f"Reading modified transcript from {modified_file}")
        with open(modified_file, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        summary_text = " ".join(entry["entire_sentence_transcript"] for entry in transcript_data)
        total_duration = sum(entry["end"] - entry["start"] for entry in transcript_data)
        troll_count = sum(1 for entry in transcript_data if entry.get("troll"))

        summary = {
            "summary_text": summary_text,
            "total_duration_seconds": total_duration,
            "sentence_count": len(transcript_data),
            "troll_clip_count": troll_count
        }

        logger.info("Summary generated")
        return {
            "status": "Summary generated successfully",
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error summarizing transcript: {str(e)}")
        return {"status": f"Error summarizing transcript: {str(e)}"}

async def create_troll(filename: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Process video input, add troll clips, and merge videos."""
    try:
        logger.info("Received video input from ADK web UI")
        filename = "sample/"+filename
        audio_file_name = f"audio/audio_{os.path.basename(filename)}.mp3"
        extract_audio(filename, audio_file_name)
        transcript_data = transcribe_audio(audio_file_name)
        transcript_data= [{'transcript': 'normally I can speak not today having a bit of a hard time this evening', 'words': [{'word': 'normally', 'start_time': 0.3, 'end_time': 1.0}, {'word': 'I', 'start_time': 1.0, 'end_time': 1.1}, {'word': 'can', 'start_time': 1.1, 'end_time': 1.1}, {'word': 'speak', 'start_time': 1.1, 'end_time': 1.7}, {'word': 'not', 'start_time': 1.7, 'end_time': 4.9}, {'word': 'today', 'start_time': 4.9, 'end_time': 5.2}, {'word': 'having', 'start_time': 5.2, 'end_time': 6.1}, {'word': 'a', 'start_time': 6.1, 'end_time': 6.1}, {'word': 'bit', 'start_time': 6.1, 'end_time': 6.3}, {'word': 'of', 'start_time': 6.3, 'end_time': 6.4}, {'word': 'a', 'start_time': 6.4, 'end_time': 6.4}, {'word': 'hard', 'start_time': 6.4, 'end_time': 6.7}, {'word': 'time', 'start_time': 6.7, 'end_time': 6.8}, {'word': 'this', 'start_time': 6.8, 'end_time': 7.2}, {'word': 'evening', 'start_time': 7.2, 'end_time': 7.3}]}, {'transcript': " yeah I'm good I'm good I love everything about you", 'words': [{'word': 'yeah', 'start_time': 11.7, 'end_time': 12.1}, {'word': "I'm", 'start_time': 12.1, 'end_time': 15.1}, {'word': 'good', 'start_time': 15.1, 'end_time': 15.4}, {'word': "I'm", 'start_time': 15.4, 'end_time': 15.5}, {'word': 'good', 'start_time': 15.5, 'end_time': 15.8}, {'word': 'I', 'start_time': 15.8, 'end_time': 19.2}, {'word': 'love', 'start_time': 19.2, 'end_time': 19.2}, {'word': 'everything', 'start_time': 19.2, 'end_time': 19.7}, {'word': 'about', 'start_time': 19.7, 'end_time': 19.9}, {'word': 'you', 'start_time': 19.9, 'end_time': 20.2}]}]
        
        logger.info(f"Transcription data: {transcript_data}")

        # Transform transcript data
        result = []
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

        # Get total video duration using ffprobe
        probe = ffmpeg.probe(filename)
        video_duration = float(probe['format']['duration'])

        # Modify transcript with troll clips
        modify_result = modify_transcript(result, tool_context)
        #modify_result = {"status": "Transcript modified with troll clips successfully","modified_file": "modified_transcript.json","modified_data": json.loads("""[{"entire_sentence_transcript": "normally I can speak not today having a bit of a hard time this evening","start": 0.3,"end": 7.3,"troll": null},{"entire_sentence_transcript": "yeah I'm good I'm good I love everything about you","start": 11.7,"end": 20.2,"troll": "4 .mp4"}]""")}
        
        logger.info(f"Modified transcript data: {modify_result}")
        if "modified_file" not in modify_result:
            return {
                "status": "Video processed but failed to modify transcript",
                "modified_file": None,
                "output_video": None
            }

        modified_transcript = modify_result["modified_data"]
        modified_file = modify_result["modified_file"]

        # Create full timeline
        timeline = []
        last_end = 0.0
        for i, entry in enumerate(modified_transcript):
            # Add segment before this transcript (if gap exists)
            if entry["start"] > last_end:
                timeline.append({
                    "start": last_end,
                    "end": entry["start"],
                    "troll": None
                })
            # Add transcript segment with troll
            timeline.append({
                "start": entry["start"],
                "end": entry["end"],
                "troll": entry.get("troll")
            })
            last_end = entry["end"]

        # Add final segment to end of video (if gap exists)
        if last_end < video_duration:
            timeline.append({
                "start": last_end,
                "end": video_duration,
                "troll": None
            })

        # Merge video with troll clips
        output_video = f"output_{os.path.basename(filename)}"
        merge_video_with_trolls(filename, timeline, output_video)

        # Summarize the modified transcript
        summary_result = await summarize_transcript(modified_file)

        return {
            "status": f"Video processed, trolls added, and summarized",
            "modified_file": modified_file,
            "output_video": output_video,
            "summary": summary_result.get("summary", {})
        }

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return {
            "status": f"Error processing video: {str(e)}",
            "modified_file": None,
            "output_video": None
        }

# Initialize services
session_service = InMemorySessionService()

# Create the agent
root_agent = LlmAgent(
    name="troll_generator",
    model="gemini-2.0-flash",
    description=(
        "This agent creates troll videos by inserting comedic clips based on transcript sentiment. "
        "It processes video input, generates transcripts, adds troll clips, merges videos, and summarizes results."
    ),
    instruction=(
        "You are a video creation agent that inserts troll clips into videos based on transcript sentiment. "
        "Process video input to extract audio, transcribe it, prompt Gemini to add troll clips, merge the video with trolls, "
        "and summarize the results. Return the status, modified transcript file, output video file, and summary."
    ),
    tools=[
        FunctionTool(create_troll)
    ]
)

# Initialize the runner
runner = Runner(
    agent=root_agent,
    app_name="video_agent_app",
    session_service=session_service
)

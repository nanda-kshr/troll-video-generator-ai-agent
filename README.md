# Troll Video Creator

A Python-based application that automatically creates entertaining troll videos by analyzing video transcripts and inserting comedic reaction clips at appropriate moments.

## Features

- Video processing and audio extraction
- Speech-to-text transcription with timestamps
- AI-powered troll clip selection based on transcript sentiment
- Automatic video merging with troll clips
- Video format normalization and quality preservation
- Summary generation of the final video

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- Google Cloud account with the following APIs enabled:
  - Speech-to-Text API
  - Cloud Storage API
  - Gemini API

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd troll-video-creator
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```
GOOGLE_API_KEY=your_google_api_key
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_APPLICATION_CREDENTIALS=location_of_service_account
```

4. Set up Google Cloud Storage:
- Create a bucket named "troll_video_gen" (or update the bucket name in the code)
- Create a folder named "troll_video_gen_audios" in the bucket

## Project Structure

```
troll_video_creator/
├── agent.py              # Main agent implementation
├── trollclips/          # Directory containing troll video clips
├── sample/              # Directory for input videos
├── audio/               # Directory for extracted audio files
├── output/              # Directory for generated videos
└── requirements.txt     # Python dependencies
```

## Usage

1. Place your input video in the `sample/` directory.

2. Run the agent:
```bash
adk web
```

3. Type video name in chat, the processed video will be saved in the `output/` directory with the prefix "output_".

## Troll Clips

The application comes with a predefined set of troll clips, each with specific emotional contexts:
- Sarcastic reactions
- Confused expressions
- Angry outbursts
- Mocking tones
- Comedic reactions
- And more...

## Output

The application generates:
1. A modified video with inserted troll clips
2. A JSON file containing the transcript with troll clip placements
3. A summary of the video processing results

## Error Handling

The application includes comprehensive error handling for:
- Video processing failures
- Audio extraction issues
- Transcription errors
- Troll clip insertion problems
- File system operations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Acknowledgments

- Google Cloud Platform for Speech-to-Text and Storage services
- FFmpeg for video processing capabilities
- Gemini API for AI-powered clip selection 
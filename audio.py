import os
import json
import time
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

load_dotenv()

# Path to the audio file
AUDIO_FILE = "C:\\Users\\Responseinformatics\\PycharmProjects\\Speech-to-Speech\\output.mp3"

API_KEY = os.getenv("DEEPGRAM_API_KEY")

def main():
    try:
        # Record the start time
        start_time = time.time()

        # STEP 1: Create a Deepgram client using the API key
        deepgram = DeepgramClient(API_KEY)

        with open(AUDIO_FILE, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        # STEP 2: Configure Deepgram options for audio analysis
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
        )

        # STEP 3: Call the transcribe_file method with the text payload and options
        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

        # STEP 4: Convert the response to JSON format
        response_json = response.to_json(indent=4)

        # STEP 5: Save the JSON response to a file
        with open("audio.json", "w") as json_file:
            json_file.write(response_json)

        # Record the end time
        end_time = time.time()

        # Calculate and print the time taken
        execution_time = end_time - start_time
        print(f"Transcription saved to audio.json. Time taken: {execution_time:.2f} seconds")

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    main()

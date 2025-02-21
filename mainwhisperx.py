import os
import re
import asyncio
import time
import tempfile
import openai
from yt_dlp import YoutubeDL
from azure.storage.blob import BlobServiceClient
import whisperx
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")

CONTAINER_NAME = "speechcontainer"
SAVEDIR = tempfile.gettempdir()

# Helper function to download audio from YouTube
async def download_audio_yt_dlp(video_url, savedir):
    try:
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        with yt_dlp.YoutubeDL() as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            audio_title = re.sub(r'[^A-Za-z0-9\s.]','', info_dict['title']).strip()
            print(f"Sanitized audio title: {audio_title}")

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(savedir, f'{audio_title}.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'noplaylist': True,
            'ffmpeg_location': r"C:\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin"  # Adjust the path for your environment
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            print("Audio downloaded successfully!")

        audio_file_path = os.path.join(savedir, f"{audio_title}.mp3")
        temp_file_path = os.path.join(savedir, f"{audio_title}.m4a")

        if os.path.exists(audio_file_path):
            print("File found:", audio_file_path)
            return audio_file_path, audio_title
        elif os.path.exists(temp_file_path):
            print("Temporary file found:", temp_file_path)
            os.rename(temp_file_path, audio_file_path)
            print("Renamed temporary file to:", audio_file_path)
            return audio_file_path, audio_title
        else:
            print("The downloaded audio file was not found at the expected location.")
            return None

    except Exception as e:
        print("An error occurred while downloading audio:", str(e))
        return None

# Function to summarize transcription using OpenAI
async def summarize_transcription_with_openai(full_transcription, openai_api_key):
    openai.api_key = openai_api_key
    try:
        response = await asyncio.to_thread(openai.chat.completions.create,
            model="gpt-4",  # GPT-4 model
            messages=[{
                "role": "system",
                "content": "You are the best key-content provider in the world, capable of identifying key points, speaker traits, and conversation themes with high accuracy."
            }, {
                "role": "user",
                "content": f"""
                "I have a podcast transcript, and I want to extract key news, important facts, and noteworthy insights from it. Your goal is to identify information that can be published as news or shared as significant updates with the news readers. Please analyze the text and provide the following:
                1. **Major Announcements, Updates, or Breaking News**:  
                    - Identify and list any major announcements, updates, or breaking news mentioned in the podcast.  
                    - Present each item as a clear and concise bullet point.  
                2. **Important Facts**:  
                    Extract any data, statistics, or factual information that is relevant and noteworthy.  
                3. **Noteworthy Insights**:  
                    Highlight any unique perspectives, expert opinions, or thought-provoking ideas discussed in the podcast.  
                4. **Future Updates**:  
                    - Identify any mentions of upcoming events, plans, or future developments discussed in the podcast.  
                    - Present these as bullet points.  
                5. **Actionable Takeaways**:  
                    If applicable, include any actionable advice, recommendations, or calls to action mentioned in the podcast.  
                Format the output in a clear and structured way, suitable for publishing as a news article or summary. Here is the transcript: [Insert your podcast transcript here]."
                **Transcription content:**  {full_transcription}"""
            }], 
            max_tokens=1000,
            temperature=0.5
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        print("Error summarizing transcription with OpenAI:", str(e))
        return None

async def process_transcription(audio_file, title, openai_api_key):
    device = "cpu"
    compute_type = "int8"
    batch = 4
    
    start_time = time.time()
    # Transcription
    print("Loading WhisperX model...")
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    print("Starting transcription...")
    result = model.transcribe(audio, batch_size=batch)
    print("Transcription completed.")

    full_transcription = "\n".join([segment['text'] for segment in result["segments"]])
    print(full_transcription)

    # Summarize transcription
    print("Summarizing transcription...")
    summary = await summarize_transcription_with_openai(full_transcription, openai_api_key)
    if summary:
        print("Summary of the transcription:\n", summary)

        #temporary file to store the summary
        summary_file_path = os.path.join(SAVEDIR, f"{title}_summary.txt")
        with open(summary_file_path, "w") as file:
            file.write(summary)
        print(f"Summary saved to: {summary_file_path}")
        
        try:
            blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
            container_client = blob_service_client.get_container_client(CONTAINER_NAME)

            blob_client = container_client.get_blob_client(blob=f"{title}_summary.txt")

            with open(summary_file_path, "rb") as file:
                blob_client.upload_blob(file, overwrite=True)

            print(f"Summary uploaded to Azure Blob Storage: {title}_summary.txt")
        except Exception as e:
            print("Error uploading to Azure Blob Storage:", str(e))
    else:
        print("Could not summarize the transcription.")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

# Main function
async def process_audio_and_summarize(video_url):
    print("Starting main process...")

    print("Downloading audio from YouTube URL...")
    audio_file_path, title = await download_audio_yt_dlp(video_url, SAVEDIR)

    if audio_file_path:
        print(f"Audio file located at: {audio_file_path}")
        await process_transcription(audio_file_path, title, OPENAI_API_KEY)
    else:
        print("Audio file download failed. Exiting.")

import os
import subprocess
import cv2
import yt_dlp
from google.cloud import speech_v1p1beta1 as speech
import threading
import math

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/aditya/Downloads/seismic-rarity-427422-p7-ab3b4a8726ef.json"

# Configuration
USE_LOCAL_FILE = True
LOCAL_VIDEO_PATH = "/Users/aditya/Documents/code projects/theoffice/videos/ween.mp4"
# YOUTUBE_URL = "https://www.youtube.com/watch?v=96Y6mc3C1Bg"

def format_time(time_sec):
    ms = int((time_sec % 1) * 1000)
    seconds = int(time_sec)
    return f"{seconds}:{ms:03d}"

def transcribe_audio_stream(audio_url, chunk_size=30):
    """Stream audio for transcription using Google Cloud Speech-to-Text."""
    client = speech.SpeechClient()

    # Use ffmpeg to convert audio stream to raw PCM data
    ffmpeg_command = [
        "ffmpeg", "-i", audio_url, "-f", "s16le", "-ac", "1", "-ar", "16000",
        "-loglevel", "quiet", "pipe:1"
    ]
    process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    diarization_config = speech.SpeakerDiarizationConfig(enable_speaker_diarization = True)

    streaming_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        diarization_config = diarization_config
    )
    
    streaming_request = speech.StreamingRecognitionConfig(config=streaming_config, interim_results=True)

    def audio_generator():
        while True:
            data = process.stdout.read(4096)
            if not data:
                break
            yield data

    requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator())
    responses = client.streaming_recognize(config=streaming_request, requests=requests, timeout = 600)

    try:
        for response in responses:
            result = response.results[-1]
            words_info = result.alternatives[0].words
            
            current_speaker = None
            current_sentence = []
            transcription = []
            
            for word_info in words_info:
                current_end_time = word_info.start_time.total_seconds()
                if current_speaker is None:
                    current_speaker = word_info.speaker_tag
                    current_start_time = word_info.start_time.total_seconds()
                if word_info.speaker_tag != current_speaker:
                    # Speaker changed, create a new sentence
                    sentence = " ".join([word.word for word in current_sentence])
                    start_time = format_time(current_start_time)
                    end_time = format_time(current_end_time)
                    transcription.append({"speaker": current_speaker, "start": current_start_time, "end": current_end_time, "text": sentence})
                    current_speaker = word_info.speaker_tag
                    current_sentence = [word_info]
                    current_start_time = word_info.start_time.total_seconds()
                else:
                    # Same speaker, add to current sentence
                    if current_end_time - current_start_time < chunk_size:
                        current_sentence.append(word_info)
                    # if one person talking for > 10 seconds send to new chunk
                    else: 
                        sentence = " ".join([word.word for word in current_sentence])
                        start_time = format_time(current_start_time)
                        end_time = format_time(current_end_time)
                        transcription.append({"speaker": current_speaker, "start": current_start_time, "end": current_end_time, "text": sentence})
                        current_sentence = [word_info]
                        current_start_time = word_info.start_time.total_seconds()
    
    # Add the last sentence if there's anything
        if current_sentence:
            sentence = " ".join([word.word for word in current_sentence])
            start_time = format_time(current_start_time)
            end_time = format_time(current_end_time)
            transcription.append({"speaker": current_speaker, "start": current_start_time, "end": current_end_time, "text": sentence})
        
        print("\n--- Raw Transcripts ---")
        print(transcription)

    
        buckets = group_transcripts_by_time(transcription, window_size=chunk_size)

        return buckets  # drop through after first batch


    except Exception as e:
        print(f"Transcription error: {e}")
    finally:
        process.terminate()

def group_transcripts_by_time(transcripts, window_size=30):
    """
    Group transcript entries into fixed-size time chunks.

    Args:
        transcripts (list of dict): each dict must have:
            - 'start' (float): start time in seconds
            - 'end'   (float): end time in seconds
            - any other fields you need (e.g., 'speaker', 'text')
        window_size (float): chunk duration in seconds (default 10)

    Returns:
        dict[int, list[dict]]: mapping chunk index → list of transcript dicts
            Chunk 0 covers [first_window_start, first_window_start + window_size),
            chunk 1 covers [first_window_start + window_size, …), etc.
    """
    if not transcripts:
        return {}

    # 1. Determine overall span
    min_start = min(t["start"] for t in transcripts)
    max_end   = max(t["end"]   for t in transcripts)

    # 2. Align to multiples of window_size
    first_window_start = math.floor(min_start / window_size) * window_size
    last_window_end    = math.ceil (max_end   / window_size) * window_size

    # 3. Number of chunks
    total_chunks = int((last_window_end - first_window_start) / window_size)

    # 4. Prepare output dict with empty lists
    chunks = {i: [] for i in range(total_chunks)}

    # 5. Assign each transcript to every chunk it overlaps
    for t in transcripts:
        # compute offset times relative to first_window_start
        start_offset = t["start"] - first_window_start
        end_offset   = t["end"]   - first_window_start

        # compute chunk indices
        start_idx = int(math.floor(start_offset / window_size))
        end_idx   = int(math.floor((end_offset - 1e-9) / window_size))

        # clamp indices to valid range
        start_idx = max(0, min(start_idx, total_chunks - 1))
        end_idx   = max(0, min(end_idx,   total_chunks - 1))

        for idx in range(start_idx, end_idx + 1):
            chunks[idx].append(t)

    return chunks

def play_video_with_transcription(video_src, audio_src):
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Press 'q' to quit.")

    # start transcription thread
    t = threading.Thread(target=transcribe_audio_stream, args=(audio_src,))
    t.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    t.join()

if __name__ == "__main__":
    if USE_LOCAL_FILE:
        play_video_with_transcription(LOCAL_VIDEO_PATH, LOCAL_VIDEO_PATH)
    else:
        # fetch YouTube video/audio URLs
        with yt_dlp.YoutubeDL({"format": "best", "quiet": True}) as ydl:
            info = ydl.extract_info(YOUTUBE_URL, download=False)
            vid_url = info["url"]
        with yt_dlp.YoutubeDL({"format": "bestaudio/best", "quiet": True}) as ydl:
            info = ydl.extract_info(YOUTUBE_URL, download=False)
            aud_url = info["url"]
        play_video_with_transcription(vid_url, aud_url)
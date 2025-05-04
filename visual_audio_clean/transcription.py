import os
import subprocess
import cv2
import math
import threading
from google.cloud import speech_v1p1beta1 as speech
import time

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/rgopalam/Desktop/seismic-rarity-427422-p7-ab3b4a8726ef.json"

# Constants
LOCAL_VIDEO_PATH = None  # Will be set at runtime

def format_time(time_sec):
    ms = int((time_sec % 1) * 1000)
    seconds = int(time_sec)
    return f"{seconds}:{ms:03d}"
    
def transcribe_audio_stream(video_path, chunk_size=None):
    """
    Extract audio from a video and transcribe it using Google Cloud Speech-to-Text.
    
    Args:
        video_path: Path to the video file
        chunk_size: (Optional) Size of chunks in seconds for processing.
                   If None, process the entire video in one go.
                   
    Returns:
        A list of transcription sections, or a single transcription if chunk_size is None
    """
    start_time = time.time()
    print(f"Starting audio transcription for {video_path}")
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return [] if chunk_size is not None else ""
    
    client = speech.SpeechClient()

    # Use ffmpeg to convert audio stream to raw PCM data
    ffmpeg_command = [
        "ffmpeg", "-i", video_path, "-f", "s16le", "-ac", "1", "-ar", "16000",
        "-loglevel", "quiet", "pipe:1"
    ]
    process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    diarization_config = speech.SpeakerDiarizationConfig(enable_speaker_diarization=True)

    streaming_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        diarization_config=diarization_config
    )
    
    streaming_request = speech.StreamingRecognitionConfig(config=streaming_config, interim_results=True)

    def audio_generator():
        while True:
            data = process.stdout.read(4096)
            if not data:
                break
            yield data

    requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator())
    
    try:
        responses = client.streaming_recognize(config=streaming_request, requests=requests, timeout=600)
        
        current_speaker = None
        current_sentence = []
        transcription = []
        
        for response in responses:
            if not response.results:
                continue
                
            result = response.results[-1]
            if not result.alternatives:
                continue
                
            if not hasattr(result.alternatives[0], 'words'):
                continue
                
            words_info = result.alternatives[0].words
            
            for word_info in words_info:
                current_end_time = word_info.start_time.total_seconds()
                if current_speaker is None:
                    current_speaker = word_info.speaker_tag
                    current_start_time = word_info.start_time.total_seconds()
                if word_info.speaker_tag != current_speaker:
                    # Speaker changed, create a new sentence
                    sentence = " ".join([word.word for word in current_sentence])
                    start_time_fmt = format_time(current_start_time)
                    end_time_fmt = format_time(current_end_time)
                    transcription.append({
                        "start": current_start_time, 
                        "end": current_end_time, 
                        "text": sentence,
                        "speaker": current_speaker
                    })
                    current_speaker = word_info.speaker_tag
                    current_sentence = [word_info]
                    current_start_time = word_info.start_time.total_seconds()
                else:
                    # Same speaker, add to current sentence
                    sentence_threshold = 30 if chunk_size is None else chunk_size
                    if current_end_time - current_start_time < sentence_threshold:
                        current_sentence.append(word_info)
                    # if one person talking for > threshold seconds send to new chunk
                    else: 
                        sentence = " ".join([word.word for word in current_sentence])
                        start_time_fmt = format_time(current_start_time)
                        end_time_fmt = format_time(current_end_time)
                        transcription.append({
                            "start": current_start_time, 
                            "end": current_end_time, 
                            "text": sentence,
                            "speaker": current_speaker
                        })
                        current_sentence = [word_info]
                        current_start_time = word_info.start_time.total_seconds()
        
        # Add the last sentence if there's anything
        if current_sentence:
            sentence = " ".join([word.word for word in current_sentence])
            start_time_fmt = format_time(current_start_time)
            end_time_fmt = format_time(current_end_time)
            transcription.append({
                "start": current_start_time, 
                "end": current_end_time, 
                "text": sentence,
                "speaker": current_speaker
            })
        
        print(f"\n--- Transcription completed with {len(transcription)} segments ---")
        
        # If chunk_size is None, return a single string with all transcripts
        if chunk_size is None:
            # Sort the transcription by start time to ensure proper ordering
            transcription.sort(key=lambda x: x["start"])
            
            # Join all text into a single string
            full_transcript = " ".join([t["text"] for t in transcription])
            
            end_time = time.time()
            print(f"Transcription completed in {end_time - start_time:.2f} seconds")
            
            return full_transcript
        else:
            # Otherwise, group them into time-based chunks
            buckets = group_transcripts_by_time(transcription, window_size=chunk_size)
            
            end_time = time.time()
            print(f"Transcription completed in {end_time - start_time:.2f} seconds")
            
            return buckets

    except Exception as e:
        import traceback
        print(f"Transcription error: {e}")
        traceback.print_exc()
        return [] if chunk_size is not None else ""
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
        window_size (float): chunk duration in seconds (default 30)

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
    last_window_end    = math.ceil(max_end / window_size) * window_size

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
        end_idx   = max(0, min(end_idx, total_chunks - 1))

        for idx in range(start_idx, end_idx + 1):
            chunks[idx].append(t)

    return chunks

def play_video_with_transcription(video_src, audio_src):
    """Play video while simultaneously transcribing audio."""
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
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    t.join()

# For testing purposes
if __name__ == "__main__":
    video_path = "/Users/aditya/Documents/code projects/theoffice/videos/clip4.mp4"
    # Test full video transcription
    transcript = transcribe_audio_stream(video_path)
    print("\nFull Transcript:")
    print(transcript)
    
    # Test chunked transcription
    chunked_transcript = transcribe_audio_stream(video_path, chunk_size=30)
    print("\nChunked Transcript:")
    for chunk_idx, chunk_data in chunked_transcript.items():
        print(f"Chunk {chunk_idx}: {len(chunk_data)} segments")
        for segment in chunk_data[:2]:  # Print first 2 segments of each chunk
            print(f"  {segment['start']}-{segment['end']}: {segment['text'][:50]}...")
        if len(chunk_data) > 2:
            print(f"  ... and {len(chunk_data) - 2} more segments")
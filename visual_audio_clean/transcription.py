import os
import subprocess
import cv2
import yt_dlp
from google.cloud import speech
import threading

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/aditya/Downloads/seismic-rarity-427422-p7-ab3b4a8726ef.json"

# Configuration
USE_LOCAL_FILE = True
LOCAL_VIDEO_PATH = "/path/to/local/video.mp4"
YOUTUBE_URL = "https://www.youtube.com/watch?v=96Y6mc3C1Bg"

def format_time(time_sec):
    ms = int((time_sec % 1) * 1000)
    seconds = int(time_sec)
    return f"{seconds}:{ms:03d}"

def transcribe_audio_stream(source) -> list[str]:
    """Extract audio from `source` (file or URL), speaker-diarize and return list of lines."""
    client = speech.SpeechClient()

    ffmpeg_cmd = [
        "ffmpeg", "-i", source,
        "-f", "s16le", "-ac", "1", "-ar", "16000",
        "-loglevel", "quiet", "pipe:1"
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    diar_cfg = speech.SpeakerDiarizationConfig(enable_speaker_diarization=True)
    recog_cfg = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        diarization_config=diar_cfg
    )

    # 3) Build StreamingRecognitionConfig (only config + interim_results)
    stream_cfg = speech.StreamingRecognitionConfig(
        config=recog_cfg,
        interim_results=True
    )

    def gen():
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            yield chunk

    requests = (
        speech.StreamingRecognizeRequest(audio_content=chunk)
        for chunk in gen()
    )
    responses = client.streaming_recognize(
        stream_cfg,
        requests
    )

    lines = []
    try:
        current_speaker = None
        current_sentence = []
        current_start = None

        for resp in responses:
            result = resp.results[-1]
            for w in result.alternatives[0].words:
                tag = w.speaker_tag
                start = w.start_time.total_seconds()
                word = w.word

                if current_speaker is None:
                    current_speaker = tag
                    current_start = start

                if tag != current_speaker:
                    sent = " ".join(current_sentence)
                    lines.append(f"speaker {current_speaker} @ {format_time(current_start)} {sent}")
                    current_speaker = tag
                    current_sentence = [word]
                    current_start = start
                else:
                    current_sentence.append(word)

        if current_sentence:
            sent = " ".join(current_sentence)
            lines.append(f"speaker {current_speaker} @ {format_time(current_start)} {sent}")

    finally:
        proc.terminate()

    return lines

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

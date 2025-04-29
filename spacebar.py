# Example command
# if u have m1/2/3/4 mac chip use: python yolo_interactive.py "videos/ween.mp4" --output_json "labels.json" --yolo_model "yolov8n.pt" --device "mps"
# otherwise: python yolo_interactive.py "videos/ween.mp4" --output_json "labels.json" --yolo_model "yolov8n.pt"

# !pip install ultralytics opencv-python pydub numpy scipy sounddevice # Ensure necessary libraries are installed
# !pip install torch torchvision torchaudio # Dependencies for ultralytics/YOLO

import cv2
import json
import time
import argparse
import numpy as np
import os
import sys # For status messages in audio callback
import traceback
import threading
from collections import defaultdict
import torch # Often needed for YOLO models

# Third-party libraries
from ultralytics import YOLO
from pydub import AudioSegment
import sounddevice as sd
import queue # For audio data streaming

# --- Constants ---
DEFAULT_YOLO_MODEL = "yolov8n.pt"
DEFAULT_TRACKER_CONFIG = 'bytetrack.yaml'

# --- Configuration Flag ---
SHOW_ONLY_PERSONS = True # <<< SET THIS TO True or False >>>

# --- UI Styling Constants ---
INSTRUCTION_AREA_HEIGHT = 240
THUMBNAIL_AREA_Y_START_OFFSET = 135
THUMBNAIL_SIZE = 64
THUMBNAIL_PADDING = 10
THUMBNAIL_LABEL_OFFSET_Y = 12
UI_BG_COLOR = (30, 30, 30)
UI_TEXT_COLOR = (230, 230, 230)
UI_SEPARATOR_COLOR = (80, 80, 80)
UI_FONT = cv2.FONT_HERSHEY_SIMPLEX
UI_FONT_SCALE = 0.55
UI_FONT_THICKNESS = 1
UI_PADDING_X = 20
UI_PADDING_Y = 25
UI_LINE_HEIGHT = 22
CLICK_HIT_COLOR = (0, 255, 255) # Cyan highlight
FACE_CLICK_REGION_COLOR = (255, 165, 0) # Orange for face click area

# --- Global State ---
# clicked_id_state stores {'id': int, 'location': 'face'|'body'|'full_box'} or None
clicked_id_state = [None]
audio_player = None # Holds the audio player instance

# --- Audio Player Class (using sounddevice) ---
# (AudioPlayer class remains exactly the same)
class AudioPlayer:
    """Handles audio playback with pause/resume functionality using sounddevice."""
    def __init__(self, audio_segment: AudioSegment):
        self.audio_segment = audio_segment
        self.stream = None
        self.playback_thread = None
        self.q = queue.Queue()
        self.current_frame = 0
        self.paused = True
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.samplerate = audio_segment.frame_rate
        self.channels = audio_segment.channels
        self.audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / (2**(audio_segment.sample_width * 8 - 1))
        if self.channels > 1:
            self.audio_data = self.audio_data.reshape(-1, self.channels)
        print(f"AudioPlayer initialized: {len(self.audio_data)/self.samplerate:.2f}s, {self.samplerate} Hz, {self.channels} channels")

    def _stream_callback(self, outdata, frames, time, status):
        if status: print(f"Audio Stream Status: {status}", file=sys.stderr)
        with self.lock:
            if self.paused or self.stop_event.is_set(): outdata.fill(0); return
            remaining_frames = len(self.audio_data) - self.current_frame
            chunk_size = min(frames, remaining_frames)
            if chunk_size > 0:
                 outdata[:chunk_size] = self.audio_data[self.current_frame : self.current_frame + chunk_size]
                 self.current_frame += chunk_size
                 if chunk_size < frames: outdata[chunk_size:].fill(0)
            else: outdata.fill(0)

    def _playback_loop(self):
        try:
            self.stream = sd.OutputStream(samplerate=self.samplerate, channels=self.channels, callback=self._stream_callback, blocksize=1024)
            with self.stream: self.stop_event.wait()
        except Exception as e: print(f"Error in audio playback thread: {e}"); traceback.print_exc()
        finally:
             if self.stream: 
                try: self.stream.close(); 
                except: pass
             self.stream = None

    def seek(self, time_sec: float):
        with self.lock:
            target_frame = int(time_sec * self.samplerate)
            self.current_frame = max(0, min(target_frame, len(self.audio_data) - 1))

    def play(self):
        with self.lock:
            if self.paused:
                self.paused = False
                if self.playback_thread is None or not self.playback_thread.is_alive():
                    self.stop_event.clear()
                    self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
                    self.playback_thread.start()

    def pause(self):
        with self.lock:
            if not self.paused: self.paused = True

    def stop(self):
        self.stop_event.set()
        with self.lock: self.paused = True
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)
            if self.playback_thread.is_alive(): print("Warning: Audio thread did not terminate cleanly.")
        self.playback_thread = None; self.stream = None

    def get_current_time(self) -> float:
        with self.lock: return self.current_frame / self.samplerate


# --- Mouse Callback Function (Operates on Filtered Tracks) ---
def mouse_callback(event, x, y, flags, param):
    """Handles clicks on the tracks passed to it (already filtered if needed)."""
    global clicked_id_state
    if event == cv2.EVENT_LBUTTONDOWN:
        # Receives potentially pre-filtered tracks
        tracks_in_frame = param.get('tracks_in_frame', [])
        is_paused = param.get('is_paused', False)

        if not is_paused: return

        clicked_info = None
        min_dist_sq = float('inf')

        for track in tracks_in_frame: # Iterate through the provided tracks
            track_id = track['id']
            x1, y1, x2, y2 = map(int, track['bbox'])
            label = track.get('label', '') # Check label for face/body logic
            is_person = (label == 'person') # Still relevant for face logic

            click_location = "outside"

            if x1 <= x < x2 and y1 <= y < y2:
                if is_person: # Apply face/body logic only if it IS a person
                    face_h = int((y2 - y1) * 0.25)
                    face_y2 = y1 + face_h
                    click_location = "face" if y < face_y2 else "body"
                else: # If SHOW_ONLY_PERSONS is False, this handles other object types
                    click_location = "full_box"

                center_x = (x1 + x2) / 2; center_y = (y1 + y2) / 2
                dist_sq = (x - center_x)**2 + (y - center_y)**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    clicked_info = {'id': track_id, 'location': click_location}

        if clicked_info is not None:
            clicked_id_state[0] = clicked_info
            # print(f"Debug: Click registered for ID {clicked_info['id']} ({clicked_info['location']}) at ({x},{y})")


# --- Pre-processing: Detect and Track Objects using YOLO ---
# (preprocess_video_tracking function remains exactly the same)
def preprocess_video_tracking(video_path, yolo_model_name=DEFAULT_YOLO_MODEL, tracker_config=DEFAULT_TRACKER_CONFIG, device='cpu'):
    """
    Processes the entire video to detect and track objects using YOLOv8.
    (Identical to the previous version)
    """
    print(f"--- Starting Pre-processing: Tracking with {yolo_model_name} and {tracker_config} on {device} ---")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Error: Could not open video file: {video_path}"); return None, None, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps and fps > 1e-2 else 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Info: {width}x{height}, ~{total_frames} frames, FPS: {fps:.2f}")

    effective_device = device
    if device.lower() != 'cpu':
         if device.lower().startswith('cuda') and not torch.cuda.is_available(): print(f"Warning: Requested device '{device}' but CUDA not available. Falling back to CPU."); effective_device = 'cpu'
         elif device.lower() == 'mps' and not getattr(torch.backends, 'mps', None) or not torch.backends.mps.is_available(): print(f"Warning: Requested device '{device}' but MPS not available. Falling back to CPU."); effective_device = 'cpu'
         elif device.lower() == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): pass
         else:
              if device.lower() == 'mps': print(f"Warning: MPS explicitly requested but not available. Falling back to CPU."); effective_device = 'cpu'

    print(f"Loading YOLO model '{yolo_model_name}' onto device '{effective_device}'...")
    try: model = YOLO(yolo_model_name); model.to(effective_device)
    except Exception as e: print(f"Error loading YOLO model: {e}"); traceback.print_exc(); cap.release(); return None, None, 0, fps

    all_tracking_data = defaultdict(list); last_known_crops = {}
    frame_idx = 0; start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None: break
        try:
            results = model.track(frame, persist=True, tracker=tracker_config, verbose=False, device=effective_device)
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy(); track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy(); cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                names = results[0].names
                current_frame_tracks = []
                for i, track_id in enumerate(track_ids):
                    x1, y1, x2, y2 = boxes[i]; conf = confs[i]; cls_id = cls_ids[i]
                    label = names.get(cls_id, f"CLS_{cls_id}")
                    track_info = {'id': int(track_id), 'bbox': [float(b) for b in boxes[i]], 'conf': float(conf), 'cls': int(cls_id), 'label': str(label)}
                    current_frame_tracks.append(track_info)
                    ix1, iy1, ix2, iy2 = map(int, [max(0, x1), max(0, y1), min(width, x2), min(height, y2)])
                    if ix2 > ix1 and iy2 > iy1:
                        crop = frame[iy1:iy2, ix1:ix2]
                        if crop.size > 0: last_known_crops[int(track_id)] = crop
                all_tracking_data[frame_idx] = current_frame_tracks
        except Exception as e: print(f"\nError during tracking on frame {frame_idx}: {e}")
        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - start_time; fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            eta_sec = ((total_frames - frame_idx) / fps_proc) if fps_proc > 0 and total_frames > 0 else 0
            eta_min = eta_sec / 60
            print(f"  Processed frame {frame_idx}/{total_frames} ({fps_proc:.1f} FPS). Est. time remaining: {eta_min:.1f} mins")

    end_time = time.time(); cap.release()
    print(f"--- Pre-processing finished in {end_time - start_time:.2f} seconds ---")
    print(f"Tracked objects across {frame_idx} frames.")
    actual_processed_frames = frame_idx
    return dict(all_tracking_data), last_known_crops, actual_processed_frames, fps


# --- Draw Tracked Objects (Operates on Filtered Tracks) ---
def draw_tracked_objects(frame, tracks_in_frame, highlight_id=None, draw_face_region=True):
    """ Draws boxes/labels for the tracks provided (already filtered if needed). """
    color_box = (0, 255, 0); color_box_highlight = CLICK_HIT_COLOR
    color_bg_text = (0, 0, 255); color_text = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.6; thickness = 1
    fh, fw = frame.shape[:2]

    for track in tracks_in_frame: # Iterate provided tracks
        track_id = track['id']
        x1, y1, x2, y2 = map(int, track['bbox'])
        label_name = track.get('label', '')
        is_person = (label_name == 'person') # Still needed for face logic

        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(fw - 1, x2), min(fh - 1, y2)
        box_color = color_box_highlight if track_id == highlight_id else color_box

        if x2 > x1 and y2 > y1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            # Draw face region only if it's a person
            if draw_face_region and is_person:
                 face_h = int((y2 - y1) * 0.25); face_y2 = y1 + face_h
                 cv2.rectangle(frame, (x1, y1), (x2, face_y2), FACE_CLICK_REGION_COLOR, 1)

            display_label = f"ID:{track_id}"
            # Only add label name if it's a person or if we are showing all types
            if is_person or not SHOW_ONLY_PERSONS:
                 if label_name: display_label += f" {label_name}"

            (tw, th), _ = cv2.getTextSize(display_label, font, font_scale, thickness)
            text_y_base = y1 - 5; bg_y1 = text_y_base - th - 2; bg_y2 = text_y_base + 3
            if bg_y1 < 0: text_y_base = y1 + th + 2; bg_y1 = y1; bg_y2 = text_y_base + 3
            bg_x1, bg_x2 = x1, x1 + tw + 4
            bg_x1, bg_y1 = max(0, bg_x1), max(0, bg_y1); bg_x2, bg_y2 = min(fw - 1, bg_x2), min(fh - 1, bg_y2)
            if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                 cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color_bg_text, -1)
                 cv2.putText(frame, display_label, (x1 + 2, text_y_base), font, font_scale, color_text, thickness, cv2.LINE_AA)


# --- Draw Thumbnails (Operates on Filtered IDs) ---
def draw_thumbnails(canvas, last_known_crops, displayed_track_ids, start_y, canvas_w):
    """Draws thumbnails for the track IDs provided (already filtered if needed)."""
    thumb_x = UI_PADDING_X; thumb_y = start_y
    max_thumbs_per_row = max(1, (canvas_w - 2 * UI_PADDING_X) // (THUMBNAIL_SIZE + THUMBNAIL_PADDING))
    count = 0
    # displayed_track_ids is already filtered based on SHOW_ONLY_PERSONS flag
    sorted_displayed_ids = sorted(list(displayed_track_ids))

    for track_id in sorted_displayed_ids:
        if count >= max_thumbs_per_row:
            thumb_y += THUMBNAIL_SIZE + THUMBNAIL_PADDING + THUMBNAIL_LABEL_OFFSET_Y + 10
            thumb_x = UI_PADDING_X; count = 0

        crop = last_known_crops.get(track_id) # Get crop using the ID
        if crop is not None and crop.size > 0:
            try:
                thumb = cv2.resize(crop, (THUMBNAIL_SIZE, THUMBNAIL_SIZE), interpolation=cv2.INTER_AREA)
                thumb_y1, thumb_y2 = thumb_y, thumb_y + THUMBNAIL_SIZE
                thumb_x1, thumb_x2 = thumb_x, thumb_x + THUMBNAIL_SIZE
                if thumb_y2 < canvas.shape[0]:
                    cv2.rectangle(canvas, (thumb_x1 - 1, thumb_y1 - 1), (thumb_x2 + 1, thumb_y2 + 1), UI_SEPARATOR_COLOR, 1)
                    canvas[thumb_y1:thumb_y2, thumb_x1:thumb_x2] = thumb
                    label = f"ID: {track_id}"
                    (tw, th), _ = cv2.getTextSize(label, UI_FONT, 0.5, UI_FONT_THICKNESS)
                    label_x = thumb_x1 + max(0, (THUMBNAIL_SIZE - tw) // 2)
                    label_y = thumb_y2 + THUMBNAIL_LABEL_OFFSET_Y
                    if label_y + th < canvas.shape[0]:
                         cv2.putText(canvas, label, (label_x, label_y), UI_FONT, 0.5, UI_TEXT_COLOR, UI_FONT_THICKNESS, cv2.LINE_AA)
                    thumb_x += THUMBNAIL_SIZE + THUMBNAIL_PADDING; count += 1
                else: break
            except Exception as e: thumb_x += THUMBNAIL_SIZE + THUMBNAIL_PADDING; count += 1
        else: thumb_x += THUMBNAIL_SIZE + THUMBNAIL_PADDING; count += 1


# --- Get Video Information (Unchanged) ---
def get_video_info(cap):
    fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps and fps > 1e-2 else 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames and total_frames > 0:
        total_frames = int(total_frames); duration = total_frames / fps
    else: duration = 0; total_frames = -1; print("Warning: Could not determine video duration/frame count.")
    return fps, total_frames, duration, w, h

# --- Helper function to save labels safely (Unchanged) ---
def save_labels(labels, output_json):
    if not labels: print("No labels to save."); return False
    print(f"Attempting to save {len(labels)} labels to {output_json}...")
    try:
        output_dir = os.path.dirname(output_json);
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
        temp_json_path = output_json + ".tmp"
        with open(temp_json_path, 'w') as f: json.dump(labels, f, indent=2, default=str)
        os.replace(temp_json_path, output_json)
        print(f"Successfully saved labels to {output_json}"); return True
    except Exception as e:
        print(f"Error saving JSON file: {e}"); traceback.print_exc()
        if os.path.exists(temp_json_path): 
            try: os.remove(temp_json_path); 
            except Exception: pass
        return False

# --- Main Labeling Function (Handles Filtering) ---
def run_labeler_interactive(
    video_path: str,
    output_json: str,
    yolo_model: str = DEFAULT_YOLO_MODEL,
    tracker_config: str = DEFAULT_TRACKER_CONFIG,
    device: str = 'cpu'
):
    global clicked_id_state, audio_player

    # --- Pre-processing ---
    tracking_data, last_known_crops, actual_frames, video_fps = preprocess_video_tracking(
        video_path, yolo_model, tracker_config, device
    )
    if tracking_data is None: print("Pre-processing failed."); return

    # --- Initialization ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Error: Could not re-open video: {video_path}"); return

    label_fps, label_total_frames, duration, w, h = get_video_info(cap)
    if actual_frames > 0: label_total_frames = actual_frames; label_fps = video_fps; duration = label_total_frames / label_fps if label_fps > 0 else 0
    if duration <= 0: print("Error: Video duration invalid."); cap.release(); return
    print(f"Labeling Frame Info: {w}x{h}, {label_total_frames} frames, FPS: {label_fps:.2f}, Duration: {duration:.2f}s")
    print(f"Settings: YOLO='{yolo_model}', Tracker='{tracker_config}', ShowOnlyPersons={SHOW_ONLY_PERSONS}")

    # --- Load Audio ---
    try:
        audio_segment = AudioSegment.from_file(video_path)
        audio_player = AudioPlayer(audio_segment)
    except Exception as e: audio_player = None; print(f"Warning: Could not load audio: {e}")

    # --- Window Setup ---
    window_name = f"Video Labeling Tool ({'Persons Only' if SHOW_ONLY_PERSONS else 'All Objects'})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    canvas_h = h + INSTRUCTION_AREA_HEIGHT; canvas_w = w
    initial_win_w = min(canvas_w, 1400); initial_win_h = min(canvas_h, 800 + (INSTRUCTION_AREA_HEIGHT - 110))
    cv2.resizeWindow(window_name, initial_win_w, initial_win_h)

    # --- Loop Variables ---
    labels = []; current_frame_index = 0; is_paused = True
    quit_flag = False; input_buffer = ""; highlighted_id_on_frame = None

    # --- Helper: Resume Playback ---
    def resume_playback(current_frame_idx, fps):
        nonlocal is_paused, input_buffer, highlighted_id_on_frame
        is_paused = False; print("Auto-resuming playback...")
        if audio_player:
            current_time_sec = current_frame_idx / fps if fps > 0 else 0
            audio_player.seek(current_time_sec); audio_player.play()
        input_buffer = ""; highlighted_id_on_frame = None

    # --- Main Loop ---
    try:
        while current_frame_index < label_total_frames and not quit_flag:
            frame_start_time = time.monotonic()
            current_time_sec = current_frame_index / label_fps if label_fps > 0 else 0

            # --- Read Frame ---
            if not is_paused:
                ret, frame = cap.read();
                if not ret or frame is None: break
                current_frame_index += 1
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
                ret, frame = cap.read()
                if not ret or frame is None: break

            # --- Prepare Canvas & Get/Filter Tracks ---
            display_canvas = np.full((canvas_h, canvas_w, 3), UI_BG_COLOR, dtype=np.uint8)
            frame_copy = frame.copy()

            all_tracks_in_current_frame = tracking_data.get(current_frame_index, [])
            # <<< FILTERING LOGIC >>>
            if SHOW_ONLY_PERSONS:
                tracks_to_display = [t for t in all_tracks_in_current_frame if t.get('label') == 'person']
            else:
                tracks_to_display = all_tracks_in_current_frame # Show all if flag is False
            displayed_track_ids = {track['id'] for track in tracks_to_display}

            # --- Draw Video Area (using filtered tracks) ---
            draw_tracked_objects(frame_copy, tracks_to_display,
                                 highlight_id=highlighted_id_on_frame if is_paused else None,
                                 draw_face_region=True) # Face region logic is inside draw func
            display_canvas[0:h, 0:w] = frame_copy
            cv2.line(display_canvas, (0, h), (canvas_w, h), UI_SEPARATOR_COLOR, 1)

            # --- Draw Info/Instructions Area ---
            text_start_y = h + UI_PADDING_Y
            status_str = f"Frame: {current_frame_index}/{label_total_frames} | Time: {current_time_sec:.2f}s | [{'PAUSED' if is_paused else 'PLAYING'}]"
            cv2.putText(display_canvas, status_str, (UI_PADDING_X, text_start_y), UI_FONT, UI_FONT_SCALE, UI_TEXT_COLOR, UI_FONT_THICKNESS, cv2.LINE_AA)

            if is_paused:
                # Use filtered IDs for display and interaction
                visible_id_text = "Visible Person IDs" if SHOW_ONLY_PERSONS else "Visible IDs"
                ids_str = f"{visible_id_text}: {sorted(list(displayed_track_ids))}"
                cv2.putText(display_canvas, ids_str, (UI_PADDING_X, text_start_y + UI_LINE_HEIGHT), UI_FONT, UI_FONT_SCALE, UI_TEXT_COLOR, UI_FONT_THICKNESS, cv2.LINE_AA)
                action_str = "ACTION: [Click Face/Body] or Enter ID + [Enter] | [N]o target | [SPACE] Manual Resume | [Q]uit"
                cv2.putText(display_canvas, action_str, (UI_PADDING_X, text_start_y + 2 * UI_LINE_HEIGHT), UI_FONT, UI_FONT_SCALE, UI_TEXT_COLOR, UI_FONT_THICKNESS, cv2.LINE_AA)
                input_display_str = f"INPUT: {input_buffer}"
                cv2.putText(display_canvas, input_display_str, (UI_PADDING_X, text_start_y + 3 * UI_LINE_HEIGHT), UI_FONT, UI_FONT_SCALE, UI_TEXT_COLOR, UI_FONT_THICKNESS, cv2.LINE_AA)
                thumbnail_area_start_y = h + THUMBNAIL_AREA_Y_START_OFFSET
                # Pass filtered IDs to thumbnails
                draw_thumbnails(display_canvas, last_known_crops, displayed_track_ids, thumbnail_area_start_y, canvas_w)
                # Pass filtered tracks to callback
                callback_data = {'tracks_in_frame': tracks_to_display, 'is_paused': True}
                cv2.setMouseCallback(window_name, mouse_callback, callback_data)
            else: # Playing
                action_str = "ACTION: [SPACE] Pause | [Q]uit"
                cv2.putText(display_canvas, action_str, (UI_PADDING_X, text_start_y + 2 * UI_LINE_HEIGHT), UI_FONT, UI_FONT_SCALE, UI_TEXT_COLOR, UI_FONT_THICKNESS, cv2.LINE_AA)
                cv2.setMouseCallback(window_name, lambda *args: None)

            # --- Display Canvas ---
            cv2.imshow(window_name, display_canvas)

            # --- Handle Input ---
            wait_ms = int(max(1, (1.0/label_fps - (time.monotonic() - frame_start_time)) * 1000)) if not is_paused else 30
            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord('q') or key == ord('Q'): quit_flag = True; break
            if key == ord(' '):
                is_paused = not is_paused
                if is_paused:
                    if audio_player: audio_player.pause()
                    input_buffer = ""; highlighted_id_on_frame = None; clicked_id_state[0] = None
                else: # Manual resume
                    if audio_player: audio_player.seek(current_time_sec); audio_player.play()
                continue

            # --- Input Handling (Paused Only) ---
            if is_paused:
                # Click Handling
                if clicked_id_state[0] is not None:
                    click_info = clicked_id_state[0]; clicked_id_state[0] = None
                    clicked_id = click_info['id']; click_location = click_info['location']
                    # Validate against the *displayed* (filtered) IDs
                    if clicked_id in displayed_track_ids:
                        choice = clicked_id; highlighted_id_on_frame = choice
                        label_info = {"label_time": round(current_time_sec, 3), "label_frame_index": current_frame_index,
                                      "zoom_target_id": choice, "click_location": click_location,
                                      "tracks_in_frame": tracks_to_display, # Save the tracks that were displayed
                                      "source": "manual_interactive_yolo", "yolo_model": yolo_model, "tracker_config": tracker_config}
                        labels.append(label_info)
                        print(f"Label recorded (Click {click_location}): ID={choice} Frame={current_frame_index}")
                        resume_playback(current_frame_index, label_fps); continue
                    else: print(f"Warning: Clicked ID {clicked_id} not in displayed tracks."); highlighted_id_on_frame = None

                # Keyboard Handling
                if key != 255: highlighted_id_on_frame = None

                if key == ord('n') or key == ord('N'):
                    choice = -1
                    label_info = {"label_time": round(current_time_sec, 3), "label_frame_index": current_frame_index,
                                  "zoom_target_id": choice, "click_location": "N/A",
                                  "tracks_in_frame": tracks_to_display, # Save the tracks that were displayed
                                  "source": "manual_interactive_yolo", "yolo_model": yolo_model, "tracker_config": tracker_config}
                    labels.append(label_info)
                    print(f"Label recorded (N): No Target Frame={current_frame_index}")
                    resume_playback(current_frame_index, label_fps); continue

                elif ord('0') <= key <= ord('9'): input_buffer += chr(key)
                elif key == 8 or key == 127: input_buffer = input_buffer[:-1]
                elif key == 13 or key == 10: # Enter
                    if input_buffer:
                        try:
                            pressed_id = int(input_buffer)
                            # Validate against the *displayed* (filtered) IDs
                            if pressed_id in displayed_track_ids:
                                choice = pressed_id; highlighted_id_on_frame = choice
                                label_info = {"label_time": round(current_time_sec, 3), "label_frame_index": current_frame_index,
                                              "zoom_target_id": choice, "click_location": "N/A",
                                              "tracks_in_frame": tracks_to_display, # Save the tracks that were displayed
                                              "source": "manual_interactive_yolo", "yolo_model": yolo_model, "tracker_config": tracker_config}
                                labels.append(label_info)
                                print(f"Label recorded (Typed): ID={choice} Frame={current_frame_index}")
                                resume_playback(current_frame_index, label_fps); continue
                            else: print(f"Error: ID '{pressed_id}' not visible. Visible: {sorted(list(displayed_track_ids))}")
                        except ValueError: print(f"Error: Invalid input '{input_buffer}'.")
                        input_buffer = ""
                    else: print("No ID typed.")

    except Exception as e: print(f"\n--- Error During Labeling: {e} ---"); traceback.print_exc()
    finally:
        print("\nLabeling finished or interrupted. Cleaning up...")
        if cap.isOpened(): cap.release()
        if audio_player: audio_player.stop()
        cv2.destroyAllWindows(); time.sleep(0.1)
        if labels: save_labels(labels, output_json)
        else: print("No labels were generated.")

# --- Main Execution Block ---
def main():
    parser = argparse.ArgumentParser(
        description="Interactive video labeling with options to show only persons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("-o", "--output_json", default="labels_output_interactive.json", help="Path to save the output JSON labels file.")
    parser.add_argument("--yolo_model", type=str, default=DEFAULT_YOLO_MODEL, help="YOLOv8 model file.")
    parser.add_argument("--tracker_config", type=str, default=DEFAULT_TRACKER_CONFIG, help="Tracker configuration file.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for YOLO inference ('cpu', 'cuda', 'mps').")
    # Optional argument to override the script flag
    parser.add_argument("--show_all", action='store_true', help="Show all detected objects (overrides SHOW_ONLY_PERSONS=True in script).")

    args = parser.parse_args()

    global SHOW_ONLY_PERSONS # Allow modification by argument
    if args.show_all:
        SHOW_ONLY_PERSONS = False
        print("Command line argument --show_all detected: Displaying ALL objects.")

    if not os.path.isfile(args.video_path): print(f"Error: Video file not found: {args.video_path}"); return

    print(f"--- Starting Interactive Video Labeler (Show Persons Only: {SHOW_ONLY_PERSONS}) ---")
    config_str = "\n".join([f"  {k}: {v}" for k, v in vars(args).items()])
    print("Configuration:\n" + config_str)
    print("--------------------------------------------------------------------")
    # Controls remain the same
    print("Controls:")
    print("  [SPACE] : Pause / Manual Resume Playback")
    print("  [Q]     : Quit")
    print("--- When Paused ---")
    print("  [Click] : Select object (face/body specific for 'person'). Auto-resumes.")
    print("  [0-9]   : Type object ID")
    print("  [Enter] : Confirm typed ID. Auto-resumes.")
    print("  [N]     : Label as 'No Target'. Auto-resumes.")
    print("--------------------------------------------------------------------")

    try:
        run_labeler_interactive(
            args.video_path, args.output_json,
            args.yolo_model, args.tracker_config, args.device
        )
    except KeyboardInterrupt: print("\nUser interrupted (Ctrl+C). Exiting.")
    finally:
        cv2.destroyAllWindows()
        if audio_player: audio_player.stop()
        print("\nLabeling tool finished.")

if __name__ == "__main__":
    main()
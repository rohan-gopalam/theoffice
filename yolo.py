# Example command
# if you have m1/2/3/4 mac chip use: python face_labeler.py "videos/sample.mp4" --output_json "face_labels.json" --yolo_model "yolov8n.pt" --device "mps"
# otherwise: python face_labeler.py "videos/sample.mp4" --output_json "face_labels.json" --yolo_model "yolov8n.pt"

import cv2
import json
import time
import argparse
import numpy as np
import os
import traceback
import threading
from collections import defaultdict
import torch 
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN

# Third-party libraries
from ultralytics import YOLO
from pydub import AudioSegment
from pydub.playback import play
# --- Face-box helper ---------------------------------------------------------
def estimate_face_bbox(body_bbox: list[float], frac: float = 0.35) -> list[int]:
    """
    Given a full-body/person bbox [x1,y1,x2,y2] return a tight-ish face box that
    hugs the top *frac* of the body box.  Works well for upright people shots.
    """
    x1, y1, x2, y2 = map(int, body_bbox)
    h = y2 - y1
    face_h = int(h * frac)
    # Clamp to image after you know the frame shape
    return [x1, y1, x2, y1 + face_h]

# --- Constants ---
# Face tracking thresholds
EMBEDDING_THRESHOLD = 0.6  # Cosine distance threshold for face matching
DBSCAN_EPS = 0.2          # DBSCAN clustering epsilon

# Model Defaults
DEFAULT_YOLO_MODEL = "yolov8n-face.pt"  # YOLOv8 face detection model
DEFAULT_TRACKER_CONFIG = 'bytetrack.yaml'  # Or 'botsort.yaml'
YOLO_CONF_THRESHOLD = 0.5  # Confidence threshold for YOLO face detection

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
CLICK_HIT_COLOR = (0, 255, 255)

# --- Global State for Mouse Click ---
clicked_id_state = [None]  # Will store the ID clicked by the user

# --- Mouse Callback Function ---
def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks to select a face track ID."""
    global clicked_id_state
    if event == cv2.EVENT_LBUTTONDOWN:
        # param should contain {'tracks_in_frame'} - which only has face tracks now
        face_tracks = param.get('tracks_in_frame', [])  # List of {'id': int, 'bbox': [x1,y1,x2,y2], ...}

        clicked_face_id = None
        for track in face_tracks:
            x1, y1, x2, y2 = map(int, track['bbox'])  # Bbox format is [x1, y1, x2, y2]

            # Check if the click (x, y) is within this face bounding box
            if x1 <= x < x2 and y1 <= y < y2:
                clicked_face_id = track['id']
                clicked_id_state[0] = clicked_face_id  # Update the shared state
                break  # Stop after finding the first match

# --- Helper function to play audio in a separate thread ---
def play_audio_threaded(segment):
    """Plays an AudioSegment in a non-blocking way."""
    if not segment: return
    try:
        play(segment)
    except Exception as e:
        pass

# --- Face Profile Management Functions ---
def update_face_profile(face_embedding, profiles, frame_idx, embedding_threshold=EMBEDDING_THRESHOLD):
    """
    Matches a face embedding to existing profiles or creates a new one.
    
    Args:
        face_embedding: NumPy array of the face embedding
        profiles: Dict {profile_id: {"embedding": emb, "frames_seen": [idx]}}
        frame_idx: Current frame index
        embedding_threshold: Cosine distance threshold for matching
        
    Returns:
        Tuple: (profile_id, updated_profiles)
    """
    matched_pid = None
    min_dist = float('inf')
    
    # Find the best match among existing profiles
    for pid, profile_data in profiles.items():
        if profile_data.get("embedding") is not None and face_embedding is not None:
            try:
                dist = cosine(profile_data["embedding"], face_embedding)
                if dist < embedding_threshold and dist < min_dist:
                    min_dist = dist
                    matched_pid = pid
            except Exception as e:
                continue
    
    if matched_pid is not None:
        # Found a match, update the profile's frame list
        profiles[matched_pid]["frames_seen"].append(frame_idx)
        return matched_pid, profiles
    else:
        # No match found, create a new profile
        new_pid = len(profiles) + 1
        profiles[new_pid] = {
            "embedding": face_embedding,
            "frames_seen": [frame_idx],
            "name": f"Person {new_pid}"  # Initial generic name
        }
        return new_pid, profiles

def recluster_profiles(profiles, eps=DBSCAN_EPS):
    """
    Recluster profile embeddings using DBSCAN based on cosine distance.
    
    Args:
        profiles: Dictionary {profile_id: {"embedding": emb, "frames_seen": [...]}}
        eps: DBSCAN epsilon distance threshold
        
    Returns:
        Tuple: (new_profile_assignments, final_profiles_summary)
    """
    print(f"\nReclustering profiles using DBSCAN with eps={eps}...")
    if not profiles:
        print("No profiles to recluster.")
        return {}, {}
    
    profile_ids = list(profiles.keys())
    embeddings = [profiles[pid]["embedding"] for pid in profile_ids if profiles[pid].get("embedding") is not None]
    valid_profile_ids = [pid for pid in profile_ids if profiles[pid].get("embedding") is not None]
    
    if not embeddings:
        print("No valid embeddings found for reclustering.")
        return {}, {}
    
    embeddings = np.array(embeddings)
    if embeddings.ndim != 2:
        print(f"Error: Embeddings array has unexpected shape {embeddings.shape}. Cannot cluster.")
        # Create a dummy mapping where each valid profile is its own cluster
        new_profile_assignments = {pid: i for i, pid in enumerate(valid_profile_ids)}
        final_profiles_summary = {
            i: {"name": profiles[pid].get("name", f"Person {i+1}"), "frames_seen": profiles[pid]["frames_seen"]}
            for i, pid in enumerate(valid_profile_ids)
        }
        return new_profile_assignments, final_profiles_summary
    
    # Use DBSCAN with cosine metric
    db = DBSCAN(eps=eps, min_samples=1, metric='cosine').fit(embeddings)
    labels = db.labels_  # Cluster labels for each embedding (-1 for noise)
    
    new_profile_assignments = {old_pid: label for old_pid, label in zip(valid_profile_ids, labels)}
    
    # Aggregate results for final summary
    final_profiles_summary = {}
    for old_pid, cluster_label in new_profile_assignments.items():
        if cluster_label == -1:  # Handle noise points if they occur (assign unique ID)
            unique_id = max(final_profiles_summary.keys()) + 1 if final_profiles_summary else 0
            final_profiles_summary[unique_id] = {
                "name": profiles[old_pid].get("name", f"Person {unique_id+1}"),
                "frames_seen": sorted(list(set(profiles[old_pid]["frames_seen"])))
            }
            new_profile_assignments[old_pid] = unique_id  # Update assignment map
            print(f"Profile {old_pid} treated as noise, assigned new ID {unique_id+1}")
            continue
        
        if cluster_label not in final_profiles_summary:
            final_profiles_summary[cluster_label] = {
                "name": f"Person {cluster_label + 1}",  # Default clustered name
                "frames_seen": []
            }
        
        # Aggregate frames
        final_profiles_summary[cluster_label]["frames_seen"].extend(profiles[old_pid]["frames_seen"])
        
        # Logic to determine the best name for the cluster
        current_name = final_profiles_summary[cluster_label]["name"]
        original_name = profiles[old_pid].get("name", "")
        if not current_name.startswith("Person ") and original_name.startswith("Person "):
            pass  # Keep the existing non-generic name
        elif original_name and not original_name.startswith("Person "):
            final_profiles_summary[cluster_label]["name"] = original_name  # Use the identified name
    
    # Clean up frame lists (sort and unique)
    for label in final_profiles_summary:
        final_profiles_summary[label]["frames_seen"] = sorted(list(set(final_profiles_summary[label]["frames_seen"])))
    
    print(f"Clustering resulted in {len(final_profiles_summary)} distinct profiles.")
    # Renumber cluster labels to be 1-based sequential IDs for final output
    final_renumbered_profiles = {}
    final_renumbered_assignments = {}
    sorted_labels = sorted(final_profiles_summary.keys())
    label_to_new_id = {label: i + 1 for i, label in enumerate(sorted_labels)}
    
    for old_pid, cluster_label in new_profile_assignments.items():
        final_renumbered_assignments[old_pid] = label_to_new_id[cluster_label]
    
    for label in sorted_labels:
        new_id = label_to_new_id[label]
        final_renumbered_profiles[new_id] = final_profiles_summary[label]
        # Update name if it's still generic and can be improved
        if final_renumbered_profiles[new_id]["name"] == f"Person {label + 1}":
            final_renumbered_profiles[new_id]["name"] = f"Person {new_id}"
    
    return final_renumbered_assignments, final_renumbered_profiles

# --- Face Detection and Embedding Extraction ---
def detect_faces_yolo(frame, yolo_model, conf_threshold=YOLO_CONF_THRESHOLD):
    """
    Detect faces in a frame using YOLO.
    
    Args:
        frame: Input frame as a NumPy array
        yolo_model: Loaded YOLO model
        conf_threshold: Confidence threshold
        
    Returns:
        List of face data dictionaries with bounding boxes and embeddings
    """
    results = yolo_model(frame)
    face_data = []
    
    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [N,4]
        confs = results[0].boxes.conf.cpu().numpy()  # [N]
        
        for i, (box, conf) in enumerate(zip(boxes, confs)):
            if conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, box)
                # Extract face region for embedding
                face_region = frame[y1:y2, x1:x2]
                
                # In a real implementation, you'd compute a face embedding here
                # For this example, we'll use a simple feature as a placeholder
                # In production, you'd use a proper face embedding model like FaceNet, ArcFace, etc.
                placeholder_embedding = np.random.rand(128)  # Placeholder 128D embedding
                
                face_data.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': float(conf),
                    'embedding': placeholder_embedding
                })
    
    return face_data

# --- Pre-processing: Detect and Track Faces using YOLO ---
def preprocess_video_face_tracking(video_path, yolo_model_name=DEFAULT_YOLO_MODEL, device='cpu'):
    """
    Processes the entire video to detect and track faces using YOLOv8.
    
    Args:
        video_path: Path to the video file
        yolo_model_name: Name of the YOLO model file
        device: Device to run inference on ('cpu', 'cuda', 'mps', etc.)
        
    Returns:
        dict: Tracking data mapping frame index to track dictionaries
        dict: Face profiles with consistent identities
        dict: Mapping of original track IDs to final profile IDs
        dict: A dictionary mapping track ID to the last good crop
        int: Total number of frames processed
        float: Video FPS
    """
    print(f"--- Starting Pre-processing: Face Tracking with {yolo_model_name} on {device} ---")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None, None, None, None, 0, 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1e-2 else 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Info: {width}x{height}, ~{total_frames} frames, FPS: {fps:.2f}")
    
    effective_device = device
    if device.lower() != 'cpu':
        if device.lower().startswith('cuda') and not torch.cuda.is_available():
            print(f"Warning: Requested device '{device}' but CUDA not available. Falling back to CPU.")
            effective_device = 'cpu'
        elif device.lower() == 'mps' and not getattr(torch.backends, 'mps', None) or not torch.backends.mps.is_available():
            print(f"Warning: Requested device '{device}' but MPS not available. Falling back to CPU.")
            effective_device = 'cpu'
    
    print(f"Loading YOLO face model '{yolo_model_name}' onto device '{effective_device}'...")
    try:
        model = YOLO(yolo_model_name)
        model.to(effective_device)  # Ensure model is on the correct device
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        traceback.print_exc()
        cap.release()
        return None, None, None, None, 0, fps
    
    all_tracking_data = defaultdict(list)  # {frame_idx: [track_info, ...]}
    face_profiles = {}  # Will store consistent face identities
    last_known_crops = {}  # {track_id: latest_crop_image}
    frame_idx = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        
        # Detect faces
        face_data = detect_faces_yolo(frame, model, conf_threshold=YOLO_CONF_THRESHOLD)
        
        current_frame_tracks = []
        for i, face in enumerate(face_data):
            # Match or create face profile
            profile_id, face_profiles = update_face_profile(
                face['embedding'], 
                face_profiles, 
                frame_idx
            )
            
            x1, y1, x2, y2 = face['bbox']
             # ⬇️ NEW — derive face box from the body/person box
            face_box = estimate_face_bbox([x1, y1, x2, y2])

            track_info = {
                'id'    : int(profile_id),
                'bbox'  : [float(x1), float(y1), float(x2), float(y2)],   # full body
                'face_bbox': list(map(float, face_box)),                  # <-- new key
                'conf'  : float(face['conf']),
                'label' : f"Face {profile_id}"
            }

            current_frame_tracks.append(track_info)
            
            # Store crop for thumbnails
            # Ensure coordinates are valid integers within frame bounds
            ix1, iy1, ix2, iy2 = map(int, [max(0, x1), max(0, y1), min(width, x2), min(height, y2)])
            if ix2 > ix1 and iy2 > iy1:  # Check for valid crop dimensions
                crop = frame[iy1:iy2, ix1:ix2]
                if crop.size > 0:
                    last_known_crops[profile_id] = crop  # Update latest crop for this ID
        
        all_tracking_data[frame_idx] = current_frame_tracks
        
        frame_idx += 1
        if frame_idx % 100 == 0:  # Print progress
            elapsed = time.time() - start_time
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            eta_sec = ((total_frames - frame_idx) / fps_proc) if fps_proc > 0 and total_frames > 0 else 0
            eta_min = eta_sec / 60
            print(f"  Processed frame {frame_idx}/{total_frames} ({fps_proc:.1f} FPS). Est. time remaining: {eta_min:.1f} mins")
    
    end_time = time.time()
    cap.release()
    print(f"--- Pre-processing finished in {end_time - start_time:.2f} seconds ---")
    print(f"Tracked faces across {frame_idx} frames.")
    
    # Recluster profiles for more consistent identities
    profile_assignments, final_profiles = recluster_profiles(face_profiles, eps=DBSCAN_EPS)
    print(f"Final face profiles after clustering: {len(final_profiles)}")
    
    # Update tracking data with new profile IDs
    for frame_idx, tracks in all_tracking_data.items():
        for track in tracks:
            old_id = track['id']
            if old_id in profile_assignments:
                new_id = profile_assignments[old_id]
                track['id'] = new_id
                track['label'] = f"Face {new_id}"
    
    # Update last_known_crops with new profile IDs
    updated_crops = {}
    for old_id, crop in last_known_crops.items():
        if old_id in profile_assignments:
            new_id = profile_assignments[old_id]
            updated_crops[new_id] = crop
    
    # ⬇️ NEW — derive face box from the body/person box
    face_box = estimate_face_bbox([x1, y1, x2, y2])

    track_info = {
        'id'    : int(profile_id),
        'bbox'  : [float(x1), float(y1), float(x2), float(y2)],   # full body
        'face_bbox': list(map(float, face_box)),                  # <-- new key
        'conf'  : float(face['conf']),
        'label' : f"Face {profile_id}"
    }

    
    return dict(all_tracking_data), final_profiles, profile_assignments, updated_crops, frame_idx, fps

# --- Draw Tracked Faces ---
def draw_tracked_faces(frame, tracks_in_frame, highlight_id=None,
                       show_body_box=True, show_face_box=True):

    """ 
    Draws bounding boxes and labels for tracked faces. Optionally highlights one ID. 
    """
    color_box = (0, 255, 0)        # Green box
    color_box_highlight = CLICK_HIT_COLOR  # Yellow box for highlight
    color_bg_text = (0, 0, 255)    # Red background for ID text
    color_text = (255, 255, 255)   # White ID text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    fh, fw = frame.shape[:2]
    
    for track in tracks_in_frame:
        track_id = track['id']
        x1, y1, x2, y2 = map(int, track['bbox'])  # Bbox is [x1, y1, x2, y2]
        label_name = track.get('label', '')  # Get face label if available
        
        # Clamp coordinates to frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw - 1, x2), min(fh - 1, y2)
        
        # Choose box color
        box_color = color_box_highlight if track_id == highlight_id else color_box
                # ---------------- BODY BOX -----------------
        if show_body_box and x2 > x1 and y2 > y1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # ---------------- FACE BOX -----------------
        if show_face_box and 'face_bbox' in track:
            fx1, fy1, fx2, fy2 = map(int, track['face_bbox'])
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2),
                          (255, 0, 0) if track_id != highlight_id else (0, 255, 255),
                          2)

        
        # Draw bounding box if valid dimensions
        if x2 > x1 and y2 > y1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Prepare text label
            display_label = f"ID:{track_id}"
            if label_name:
                display_label += f" {label_name}"
            
            (tw, th), _ = cv2.getTextSize(display_label, font, font_scale, thickness)
            
            # Position text slightly above the box
            text_y_base = y1 - 5
            bg_y1 = text_y_base - th - 2
            bg_y2 = text_y_base + 3
            if bg_y1 < 0:  # If text goes off top, put inside box near top
                text_y_base = y1 + th + 2
                bg_y1 = y1
                bg_y2 = text_y_base + 3
            
            # Background rectangle position and clamping
            bg_x1 = x1
            bg_x2 = x1 + tw + 4
            bg_x1, bg_y1 = max(0, bg_x1), max(0, bg_y1)
            bg_x2, bg_y2 = min(fw - 1, bg_x2), min(fh - 1, bg_y2)
            
            # Draw background and text if background is valid
            if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color_bg_text, -1)  # Filled bg
                cv2.putText(frame, display_label, (x1 + 2, text_y_base),
                           font, font_scale, color_text, thickness, cv2.LINE_AA)

# --- Draw Thumbnails using last known crops ---
def draw_thumbnails(canvas, last_known_crops, displayed_track_ids, start_y, canvas_w):
    """Draws thumbnails of tracked faces currently displayed, using their last known crop."""
    thumb_x = UI_PADDING_X
    thumb_y = start_y
    max_thumbs_per_row = max(1, (canvas_w - 2 * UI_PADDING_X) // (THUMBNAIL_SIZE + THUMBNAIL_PADDING))
    count = 0
    
    # Sort the IDs currently on screen for consistent thumbnail order
    sorted_displayed_ids = sorted(list(displayed_track_ids))
    
    for track_id in sorted_displayed_ids:
        if count >= max_thumbs_per_row:
            thumb_y += THUMBNAIL_SIZE + THUMBNAIL_PADDING + THUMBNAIL_LABEL_OFFSET_Y + 10
            thumb_x = UI_PADDING_X
            count = 0
        
        crop = last_known_crops.get(track_id)  # Get the last stored crop for this ID
        
        if crop is not None and crop.size > 0:
            try:
                thumb = cv2.resize(crop, (THUMBNAIL_SIZE, THUMBNAIL_SIZE), interpolation=cv2.INTER_AREA)
                thumb_y1, thumb_y2 = thumb_y, thumb_y + THUMBNAIL_SIZE
                thumb_x1, thumb_x2 = thumb_x, thumb_x + THUMBNAIL_SIZE
                
                if thumb_y2 < canvas.shape[0]:  # Ensure fits vertically
                    cv2.rectangle(canvas, (thumb_x1 - 1, thumb_y1 - 1), (thumb_x2 + 1, thumb_y2 + 1), UI_SEPARATOR_COLOR, 1)
                    canvas[thumb_y1:thumb_y2, thumb_x1:thumb_x2] = thumb
                    
                    label = f"ID: {track_id}"
                    (tw, th), _ = cv2.getTextSize(label, UI_FONT, 0.5, UI_FONT_THICKNESS)
                    label_x = thumb_x1 + max(0, (THUMBNAIL_SIZE - tw) // 2)
                    label_y = thumb_y2 + THUMBNAIL_LABEL_OFFSET_Y
                    if label_y + th < canvas.shape[0]:
                        cv2.putText(canvas, label, (label_x, label_y), UI_FONT, 0.5, UI_TEXT_COLOR, UI_FONT_THICKNESS, cv2.LINE_AA)
                    
                    thumb_x += THUMBNAIL_SIZE + THUMBNAIL_PADDING
                    count += 1
                else:
                    break  # Stop if off canvas
            except Exception as e:
                thumb_x += THUMBNAIL_SIZE + THUMBNAIL_PADDING
                count += 1
        else:
            # Crop might be missing if track just appeared or error occurred
            thumb_x += THUMBNAIL_SIZE + THUMBNAIL_PADDING
            count += 1

# --- Get Video Information ---
def get_video_info(cap):
    """ Returns (fps, total_frames, duration_sec, width, height) from cv2.VideoCapture. """
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1e-2 else 30.0  # Use default FPS if needed
    
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use frame count directly if available and > 0
    if total_frames and total_frames > 0:
        total_frames = int(total_frames)
        duration = total_frames / fps
    else:  # Fallback if frame count is unreliable
        duration = 0
        try:
            # Try seeking to get duration (less reliable)
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
            if duration_ms and duration_ms > 0:
                duration = duration_ms / 1000.0
                total_frames = int(duration * fps)  # Estimate frame count
            else: total_frames = 0; duration = 0
        except Exception:
            total_frames = 0; duration = 0
        if duration <= 0:
            print("Warning: Could not determine video duration or frame count accurately.")
            total_frames = -1  # Indicate unknown frame count
    
    return fps, total_frames, duration, w, h

# --- Helper function to save labels safely ---
def save_labels(labels, output_json):
    """Saves the current list of labels to the specified JSON file."""
    if not labels:
        print("No labels to save.")
        return False
    
    print(f"Attempting to save {len(labels)} labels to {output_json}...")
    try:
        output_dir = os.path.dirname(output_json)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)
        
        temp_json_path = output_json + ".tmp"
        with open(temp_json_path, 'w') as f:
            json.dump(labels, f, indent=2)
        os.replace(temp_json_path, output_json)  # Atomic rename
        
        print(f"Successfully saved labels to {output_json}")
        return True
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        traceback.print_exc()
        if os.path.exists(temp_json_path):
            try: os.remove(temp_json_path)
            except Exception: pass
        return False

# --- Main Labeling Function (Using Pre-processed Face Tracking) ---
def run_face_labeler(
    video_path: str,
    output_json: str,
    k_seconds: float = 2.0,
    yolo_model: str = DEFAULT_YOLO_MODEL,
    device: str = 'cpu',
    faces_only: bool = True  # Always true - we only care about faces now
):
    """
    Main loop using pre-processed YOLO face tracking results.
    Playback, pause, clickable face boxes, user input.
    """
    global clicked_id_state
    
    # --- 1. Pre-processing Step ---
    tracking_data, face_profiles, profile_assignments, last_known_crops, actual_frames, video_fps = preprocess_video_face_tracking(
        video_path, yolo_model, device
    )
    
    if tracking_data is None:
        print("Pre-processing failed. Exiting.")
        return
    
    # --- Initialization for Labeling ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not re-open video file for labeling: {video_path}")
        return
    
    # Get video info again for labeling phase dimensions/duration
    label_fps, label_total_frames, duration, w, h = get_video_info(cap)
    if duration <= 0:
        print("Error: Video duration is zero or could not be determined for labeling.")
        cap.release()
        return
    
    # Use the more accurate frame count and fps if pre-processing got them
    if actual_frames > 0:
        label_total_frames = actual_frames
        label_fps = video_fps  # Use FPS calculated during pre-processing
        duration = label_total_frames / label_fps if label_fps > 0 else 0
        print(f"Using Pre-processed Info: {w}x{h}, {label_total_frames} frames, FPS: {label_fps:.2f}, Duration: {duration:.2f}s")
    else:
        print(f"Labeling Phase Info: {w}x{h}, FPS: {label_fps:.2f}, Duration: {duration:.2f}s")
    
    print(f"Labeling Settings: YOLO Face Model='{yolo_model}', Chunk='{k_seconds}s'")
    print(f"Detected {len(face_profiles)} unique faces in the video")
    
    try:
        audio = AudioSegment.from_file(video_path)
        print(f"Audio loaded ({len(audio)/1000.0:.2f}s).")
    except Exception as e:
        audio = None
        print(f"Warning: Could not load audio: {e}")
    
    # Setup display window
    window_name = "Face Labeling Tool"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    canvas_h = h + INSTRUCTION_AREA_HEIGHT
    canvas_w = w
    cv2.resizeWindow(window_name, min(canvas_w, 1400), min(canvas_h, 900 + (INSTRUCTION_AREA_HEIGHT - 110)))
    
    # --- Main Labeling Loop Variables ---
    labels = []
    current_time_sec = 0.0
    chunk_index = 0
    quit_flag = False
    
    try:
        while current_time_sec < duration - 1e-6 and not quit_flag:
            chunk_index += 1
            chunk_start_sec = current_time_sec
            chunk_end_sec = min(chunk_start_sec + k_seconds, duration)
            print(f"\n--- Chunk #{chunk_index}: {chunk_start_sec:.2f}s -> {chunk_end_sec:.2f}s ---")
            
            # Seek to the start of the chunk in the video
            start_frame_index = int(chunk_start_sec * label_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
            
            # --- Audio Playback Thread ---
            audio_thread = None
            if audio:
                start_ms = max(0, int(chunk_start_sec * 1000))
                end_ms = min(len(audio), int(chunk_end_sec * 1000))
                if start_ms < end_ms:
                    try:
                        buffered_end_ms = min(end_ms + 150, len(audio))
                        audio_chunk = audio[start_ms:buffered_end_ms]
                        if audio_chunk and len(audio_chunk) > 20:
                            audio_thread = threading.Thread(target=play_audio_threaded, args=(audio_chunk,), daemon=True)
                            audio_thread.start()
                    except Exception as audio_err: print(f"Warning: Audio slice/play failed: {audio_err}")
            
            # --- Video Playback Loop for the Chunk ---
            last_frame_of_chunk = None
            last_frame_index_in_chunk = -1
            current_frame_index = start_frame_index
            chunk_playback_start_time = time.monotonic()
            frames_played_in_chunk = 0
            
            while True:
                # Calculate target time for smooth playback
                expected_elapsed_time = frames_played_in_chunk / label_fps
                target_wall_time = chunk_playback_start_time + expected_elapsed_time
                
                # Check if current frame index exceeds chunk end
                current_frame_time_sec = current_frame_index / label_fps if label_fps > 0 else chunk_start_sec + expected_elapsed_time
                if current_frame_time_sec >= (chunk_end_sec - (0.5 / label_fps)):  # Stop slightly before exact end time
                    break
                
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"Warning: Frame read failed at index {current_frame_index} (time {current_frame_time_sec:.2f}s)")
                    break  # End chunk if frame read fails
                
                last_frame_of_chunk = frame  # Store the last valid frame
                last_frame_index_in_chunk = current_frame_index
                frames_played_in_chunk += 1
                current_frame_index += 1  # Move to next frame index
                
                # --- Prepare Display Canvas (Playback) ---
                display_canvas = np.full((canvas_h, canvas_w, 3), UI_BG_COLOR, dtype=np.uint8)
                display_canvas[0:h, 0:w] = frame.copy()
                cv2.line(display_canvas, (0, h), (canvas_w, h), UI_SEPARATOR_COLOR, 1)
                time_str = f"Time: {current_frame_time_sec:.2f} / {duration:.2f} s (Frame: {current_frame_index})"
                (tw, th), _ = cv2.getTextSize(time_str, UI_FONT, 0.6, 1)
                cv2.rectangle(display_canvas, (10, 10), (10 + tw + 10, 10 + th + 10), (0,0,0, 128), -1)
                cv2.putText(display_canvas, time_str, (15, 15 + th), UI_FONT, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow(window_name, display_canvas)
                
                # Calculate wait time
                current_wall_time = time.monotonic()
                wait_time_sec = target_wall_time - current_wall_time
                wait_ms = max(1, int(wait_time_sec * 1000))
                
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("User pressed Q during playback -> quitting.")
                    quit_flag = True; break
            # --- End of Playback Loop ---
            
            if quit_flag: break
            
            if last_frame_of_chunk is None:
                print("Warning: No valid frames displayed in this chunk. Skipping labeling.")
                current_time_sec = chunk_end_sec
                continue
            
            # --- Pause and Prepare for Labeling ---
            print(f"Paused at end of chunk (Frame: {last_frame_index_in_chunk}, Time: {chunk_end_sec:.2f}s)")
            
            # --- Get Pre-computed Tracks for the Last Frame ---
            tracks_in_last_frame = tracking_data.get(last_frame_index_in_chunk, [])
            displayed_track_ids = {track['id'] for track in tracks_in_last_frame}
            print(f" -> Faces present in last frame: {sorted(list(displayed_track_ids))}")
            
            # --- Prepare Display Canvas (Pause/Labeling) ---
            pause_canvas = np.full((canvas_h, canvas_w, 3), UI_BG_COLOR, dtype=np.uint8)
            frame_with_boxes = last_frame_of_chunk.copy()
            
            # Draw boxes using the pre-computed tracks for this frame
            draw_tracked_faces(frame_with_boxes, tracks_in_last_frame,
                   highlight_id=None,
                   show_body_box=True,   # existing green box
                   show_face_box=True)   # new blue/yellow face box

            
            pause_canvas[0:h, 0:w] = frame_with_boxes
            cv2.line(pause_canvas, (0, h), (canvas_w, h), UI_SEPARATOR_COLOR, 1)
            
            # --- Draw Instructions ---
            text_start_y = h + UI_PADDING_Y
            chunk_str = f"CHUNK INFO: {chunk_index} ({chunk_start_sec:.2f}s - {chunk_end_sec:.2f}s)"
            cv2.putText(pause_canvas, chunk_str, (UI_PADDING_X, text_start_y), UI_FONT, UI_FONT_SCALE, UI_TEXT_COLOR, UI_FONT_THICKNESS, cv2.LINE_AA)
            
            ids_str_present = f"Visible Face IDs (Click or Type): {sorted(list(displayed_track_ids))}"
            cv2.putText(pause_canvas, ids_str_present, (UI_PADDING_X, text_start_y + UI_LINE_HEIGHT), UI_FONT, UI_FONT_SCALE, UI_TEXT_COLOR, UI_FONT_THICKNESS, cv2.LINE_AA)
            
            action_str = "ACTION: [Click a Face] or Enter Face ID + [Enter] | [N]o selection | [Q]uit"
            cv2.putText(pause_canvas, action_str, (UI_PADDING_X, text_start_y + 3 * UI_LINE_HEIGHT), UI_FONT, UI_FONT_SCALE, UI_TEXT_COLOR, UI_FONT_THICKNESS, cv2.LINE_AA)  # Adjusted Y
            
            input_buffer_y = text_start_y + 4 * UI_LINE_HEIGHT  # Adjusted Y
            
            # --- Draw Thumbnails ---
            thumbnail_area_start_y = h + THUMBNAIL_AREA_Y_START_OFFSET
            draw_thumbnails(pause_canvas, last_known_crops, displayed_track_ids, thumbnail_area_start_y, canvas_w)
            
            # --- Set up Mouse Callback ---
            clicked_id_state[0] = None
            callback_data = {'tracks_in_frame': tracks_in_last_frame}
            cv2.setMouseCallback(window_name, mouse_callback, callback_data)
            
            # Display initial pause screen
            cv2.imshow(window_name, pause_canvas)
            
            # --- User Input Loop ---
            print(f"Waiting for user input (Visible Face IDs: {sorted(list(displayed_track_ids))}). Click a face or type a face ID...")
            choice = None
            input_buffer = ""
            highlighted_id_on_frame = None
            
            while True:
                temp_canvas_for_input = pause_canvas.copy()  # Redraw canvas each loop
                
                if highlighted_id_on_frame is not None:
                    frame_with_highlight = last_frame_of_chunk.copy()
                    draw_tracked_faces(frame_with_boxes, tracks_in_last_frame,
                   highlight_id=highlighted_id_on_frame,
                   show_body_box=True,   # existing green box
                   show_face_box=True)   # new blue/yellow face box

                    temp_canvas_for_input[0:h, 0:w] = frame_with_highlight
                
                input_display_str = f"INPUT: {input_buffer}"
                cv2.rectangle(temp_canvas_for_input, (0, input_buffer_y - UI_LINE_HEIGHT + 5), (canvas_w, input_buffer_y + 5), UI_BG_COLOR, -1)
                cv2.putText(temp_canvas_for_input, input_display_str, (UI_PADDING_X, input_buffer_y), UI_FONT, UI_FONT_SCALE, UI_TEXT_COLOR, UI_FONT_THICKNESS, cv2.LINE_AA)
                cv2.imshow(window_name, temp_canvas_for_input)
                
                # Check for click
                if clicked_id_state[0] is not None:
                    clicked_id = clicked_id_state[0]
                    clicked_id_state[0] = None
                    # Check if the clicked ID is actually one displayed in this frame
                    if clicked_id in displayed_track_ids:
                        choice = clicked_id
                        highlighted_id_on_frame = clicked_id
                        print(f"Selected face ID = {choice} (via Click)")
                        # Redraw immediately to show highlight
                        frame_with_highlight = last_frame_of_chunk.copy()
                        draw_tracked_faces(frame_with_boxes, tracks_in_last_frame,
                   highlight_id=highlighted_id_on_frame,
                   show_body_box=True,   # existing green box
                   show_face_box=True)   # new blue/yellow face box
                        temp_canvas_for_input[0:h, 0:w] = frame_with_highlight
                        cv2.rectangle(temp_canvas_for_input, (0, input_buffer_y - UI_LINE_HEIGHT + 5), (canvas_w, input_buffer_y + 5), UI_BG_COLOR, -1)  # Clear input line
                        cv2.putText(temp_canvas_for_input, f"INPUT: [Clicked Face ID {choice}]", (UI_PADDING_X, input_buffer_y), UI_FONT, UI_FONT_SCALE, UI_TEXT_COLOR, UI_FONT_THICKNESS, cv2.LINE_AA)
                        cv2.imshow(window_name, temp_canvas_for_input)
                        cv2.waitKey(150)  # Brief pause
                        break
                    else:
                        print(f"Warning: Click detected for ID {clicked_id}, but it's not listed in this frame's faces: {sorted(list(displayed_track_ids))}.")
                        highlighted_id_on_frame = None
                
                key = cv2.waitKey(50) & 0xFF
                if key == 255: continue
                
                highlighted_id_on_frame = None  # Clear highlight on keyboard input
                
                if key == ord('q') or key == ord('Q'): quit_flag = True; break
                elif key == ord('n') or key == ord('N'): choice = -1; print("Selected: No target (N)"); break
                elif ord('0') <= key <= ord('9'): input_buffer += chr(key)
                elif key == 8 or key == 127: input_buffer = input_buffer[:-1]  # Backspace
                elif key == 13 or key == 10:  # Enter
                    if input_buffer:
                        try:
                            pressed_id = int(input_buffer)
                            # VALIDATION: Check against IDs VISIBLE in THIS frame
                            if pressed_id in displayed_track_ids:
                                choice = pressed_id
                                highlighted_id_on_frame = choice  # Highlight the selected box
                                print(f"Selected face ID = {choice} (via Keyboard)")
                                # Redraw to show highlight
                                frame_with_highlight = last_frame_of_chunk.copy()
                                draw_tracked_faces(frame_with_boxes, tracks_in_last_frame,
                   highlight_id=highlighted_id_on_frame,
                   show_body_box=True,   # existing green box
                   show_face_box=True)   # new blue/yellow face box
                                temp_canvas_for_input[0:h, 0:w] = frame_with_highlight
                                cv2.rectangle(temp_canvas_for_input, (0, input_buffer_y - UI_LINE_HEIGHT + 5), (canvas_w, input_buffer_y + 5), UI_BG_COLOR, -1)  # Clear input line
                                cv2.putText(temp_canvas_for_input, f"INPUT: [Typed Face ID {choice}]", (UI_PADDING_X, input_buffer_y), UI_FONT, UI_FONT_SCALE, UI_TEXT_COLOR, UI_FONT_THICKNESS, cv2.LINE_AA)
                                cv2.imshow(window_name, temp_canvas_for_input)
                                cv2.waitKey(150)  # Brief pause
                                break
                            else:
                                print(f"Error: ID '{pressed_id}' is not visible in this frame {sorted(list(displayed_track_ids))}. Try again or Click.")
                                input_buffer = ""
                        except ValueError:
                            print(f"Error: Invalid number '{input_buffer}'. Try again.")
                            input_buffer = ""
                    else:
                        print(f"No ID typed. Type a Visible ID {sorted(list(displayed_track_ids))}, Click Face, N, or Q.")
                # Ignore other keys
            
            # --- End of Input Loop ---
            
            cv2.setMouseCallback(window_name, lambda *args: None)  # Disable callback
            
            if quit_flag: break
            
            if choice is not None:
                # Store label with pre-computed tracks present in the *last frame* of the chunk
                label_info = {
                    "chunk_index": chunk_index,
                    "start_time": round(chunk_start_sec, 3),
                    "end_time": round(chunk_end_sec, 3),
                    "last_frame_index": last_frame_index_in_chunk,
                    "target_face_id": choice,  # The stable, pre-computed face ID selected
                    "faces_in_last_frame": tracks_in_last_frame,  # BBoxes & IDs visible at decision point
                    "source": "face_tracking_yolo",
                    "yolo_model": yolo_model
                }
                labels.append(label_info)
                print("Label recorded:", json.dumps({"chunk_index": chunk_index, "face_id": choice, "time": f"{chunk_start_sec:.2f}-{chunk_end_sec:.2f}"}))  # Compact print
            else:
                print(f"No choice recorded for chunk {chunk_index}.")
            
            # Move to the next chunk
            current_time_sec = chunk_end_sec
        
        # --- End of Main Labeling Loop ---
    
    except Exception as e:
        print("\n--- An Error Occurred During Labeling Phase ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Details: {e}")
        print("Traceback:")
        traceback.print_exc()
        print("------------------------------------")
    
    finally:
        print("\nLabeling finished or interrupted. Cleaning up...")
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        
        if labels:
            save_labels(labels, output_json)
        else:
            print("No labels were generated to save.")


# --- Main Execution Block ---
def main():
    parser = argparse.ArgumentParser(
        description="Face-focused tracking and labeling tool using YOLO with identity consistency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("-o", "--output_json", default="face_labels_output.json", help="Path to save the output JSON labels file.")
    parser.add_argument("-k", "--k_seconds", type=float, default=2.0, help="Duration of each video chunk for labeling in seconds.")
    parser.add_argument("--yolo_model", type=str, default=DEFAULT_YOLO_MODEL, help="YOLOv8 face detection model file (e.g., yolov8n-face.pt).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run YOLO inference on (e.g., 'cpu', 'cuda', 'cuda:0', 'mps').")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.video_path): print(f"Error: Video file not found: {args.video_path}"); return
    if args.k_seconds <= 0: print("Error: k_seconds (chunk duration) must be positive."); return
    
    print("--- Starting Face-Only Labeling Tool ---")
    config_str = "\n".join([f"  {k}: {v}" for k, v in vars(args).items()])
    print("Configuration:\n" + config_str)
    print("--------------------------------------------------------------------")
    
    try:
        run_face_labeler(
            args.video_path, args.output_json, args.k_seconds,
            args.yolo_model, args.device, faces_only=True
        )
    except KeyboardInterrupt:
        print("\nUser interrupted (Ctrl+C). Exiting.")
    finally:
        cv2.destroyAllWindows()  # Ensure windows close on exit
        print("\nFace labeling tool finished.")

if __name__ == "__main__":
    main()


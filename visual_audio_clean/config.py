# config.py
import torch
from torchvision import transforms
import os # <-- Add import os

# --- Device Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Input Video & Frame Extraction ---
# **** IMPORTANT: SET YOUR VIDEO PATH HERE ****
VIDEO_INPUT_PATH = "/Users/rgopalam/Desktop/office_clips/20250312_FineDrabTardigradeAMPTropPunch-v5zAsfIVNJNlkktn_source.mp4" # <--- CHANGE THIS TO YOUR VIDEO FILE
FRAMES_PER_SECOND_TO_EXTRACT = 1 # Extract 1 frame per second. Set to 0 or None for all frames.

# --- Output Directories ---
OUTPUT_DIR = "output" # Main directory for all outputs
FRAME_OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, "extracted_frames") # Subdir for frames
VISUALIZATION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "visualizations") # Subdir for final viz images
LLM_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "llm_analysis_input.json") # Path for LLM JSON

# --- Model & Analysis Parameters ---
# Transform for gaze detection model
gaze_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

# Threshold for matching face embeddings across frames (used initially)
EMBEDDING_THRESHOLD = 0.6
# Confidence threshold for YOLO face detection
YOLO_CONF_THRESHOLD = 0.5
# IoU threshold for Non-Maximum Suppression (lower = more suppression)
NMS_IOU_THRESHOLD = 0.3 # Adjust this value (e.g., 0.3 to 0.6)

# DBSCAN clustering distance threshold
DBSCAN_EPS = 0.2 # Cosine distance threshold for merging profiles
# Gaze 'in/out' threshold
GAZE_INOUT_THRESHOLD = 0.5

# --- Model Paths & Repos ---
YOLO_MODEL_PATH = "yolov8n.pt" # Or your specific YOLO face model path
GAZE_MODEL_REPO = 'fkryan/gazelle'
GAZE_MODEL_NAME = 'gazelle_dinov2_vitl14_inout'

# --- Output Settings ---
SHOW_YOLO_DETECTIONS = False # Flag to show intermediate YOLO detections per frame visually (can be slow)
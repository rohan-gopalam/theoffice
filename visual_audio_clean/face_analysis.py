# face_analysis.py
import numpy as np
import cv2
from deepface import DeepFace
from scipy.spatial.distance import cosine
from face_detection import detect_faces_yolo # Use the detection function

def get_face_embedding(face_roi):
    """Extract face embedding using DeepFace (detection skipped)."""
    try:
        # Using Facenet512 as specified, VGG-Face is default but often less accurate
        rep = DeepFace.represent(face_roi, model_name='Facenet512',
                                 detector_backend='skip', enforce_detection=False)
        # Ensure result is structured as expected
        if isinstance(rep, list) and len(rep) > 0 and isinstance(rep[0], dict) and 'embedding' in rep[0]:
             return np.array(rep[0]['embedding'])
        else:
            print(f"Unexpected embedding format from DeepFace: {rep}")
            return None
    except Exception as e:
        print(f"Embedding error for face ROI of shape {face_roi.shape}: {e}")
        return None

def match_known_face(face_emb, known_faces):
    """
    Match a face embedding to the most similar known face using cosine distance.

    Args:
        face_emb: Detected face embedding (NumPy array).
        known_faces: Dictionary {name: embedding (NumPy array)}.

    Returns:
        Tuple (name, similarity_score (cosine distance)) or (None, float('inf')) if no match.
    """
    if not known_faces or face_emb is None:
        return None, float('inf')

    best_match = None
    best_score = float('inf') # Lower is better for cosine distance

    for kname, known_emb in known_faces.items():
        if known_emb is None or face_emb.shape != known_emb.shape:
            print(f"Warning: Skipping invalid known embedding for {kname}")
            continue
        try:
            dist = cosine(known_emb, face_emb)
            if dist < best_score:
                best_score = dist
                best_match = kname
        except Exception as e:
            print(f"Error calculating cosine distance for {kname}: {e}")

    return best_match, best_score

def analyze_emotions(face_roi):
    """
    Analyze emotions for the given face ROI using DeepFace.
    Returns dominant emotion string or 'Unknown'.
    """
    if face_roi is None or face_roi.size == 0:
        print("Emotion analysis error: Received empty face ROI.")
        return "Unknown"
    try:
        # DeepFace.analyze returns a list of dictionaries
        analysis = DeepFace.analyze(face_roi, actions=['emotion'],
                                    detector_backend='skip', enforce_detection=False,
                                    silent=True) # Add silent=True to reduce console noise
        # Check if analysis was successful and has the expected structure
        if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
            return analysis[0].get('dominant_emotion', "Unknown")
        else:
             print(f"Unexpected emotion analysis result format: {analysis}")
             return "Unknown"
    except ValueError as ve:
        # Catch specific value errors often related to face not found by internal checks
        # even with skip detector, possibly due to ROI size/quality
        # print(f"Emotion analysis ValueError (likely face quality/size issue): {ve}")
        return "Undetected" # Or return "Unknown"
    except Exception as e:
        print(f"Emotion analysis general error: {e}")
        return "Unknown"


def load_known_faces(image_paths_dict, yolo_face_model):
    """
    Load known faces (profiles) using YOLO for detection and DeepFace for embeddings.

    Args:
        image_paths_dict: Dictionary {name: image_path}
        yolo_face_model: Loaded YOLO model.

    Returns:
        Dictionary {name: embedding (NumPy array)} for known people.
    """
    known_faces = {}
    print("Loading known faces (profiles)...")
    for name, path in image_paths_dict.items():
        try:
            img = cv2.imread(path)
            if img is None:
                print(f"Could not load image for {name} at {path}")
                continue

            # Detect faces using YOLO
            detected_boxes = detect_faces_yolo(img, yolo_face_model)

            if detected_boxes:
                # Use the first detected face for the profile
                x1, y1, x2, y2 = detected_boxes[0]
                face_roi = img[y1:y2, x1:x2]

                if face_roi.size == 0:
                    print(f"Empty face ROI extracted for {name} from {path}")
                    continue

                embedding = get_face_embedding(face_roi)
                if embedding is not None:
                    known_faces[name] = embedding
                    print(f"Successfully loaded face embedding for {name}")
                else:
                    print(f"Could not get embedding for {name} in {path}")
            else:
                print(f"No face detected by YOLO for {name} in {path}")
        except Exception as e:
            print(f"Error loading face for {name} from {path}: {e}")
    print(f"Finished loading known faces. Found {len(known_faces)} profiles.")
    return known_faces
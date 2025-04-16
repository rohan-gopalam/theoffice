# face_detection.py
import cv2
from ultralytics import YOLO
import config  # Import configuration

def load_yolo_model(model_path=config.YOLO_MODEL_PATH, device=config.device):
    """Loads the YOLO model."""
    print(f"Loading YOLO model from {model_path} on {device}...")
    try:
        model = YOLO(model_path)
        model.to(device)
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        raise # Re-raise the exception to halt execution if loading fails

def detect_faces_yolo(img_array, yolo_face_model, conf_threshold=config.YOLO_CONF_THRESHOLD):
    """
    Detect faces in an image using YOLO.

    Args:
        img_array: Input image as a NumPy array.
        yolo_face_model: Loaded YOLO model.
        conf_threshold: Confidence threshold.

    Returns:
        List of bounding boxes [x1, y1, x2, y2] in pixel coordinates.
    """
    results = yolo_face_model(img_array)
    face_boxes = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [N,4]
        confs = results[0].boxes.conf.cpu().numpy()    # [N]
        for box, conf in zip(boxes, confs):
            if conf >= conf_threshold:
                face_boxes.append([int(x) for x in box])
    return face_boxes

def visualize_yolo_detections(img_array, yolo_face_model, save_path=None):
    """
    Use the YOLO model's built-in plotting method to create an annotated image.

    Args:
        img_array: Input image as a NumPy array.
        yolo_face_model: Loaded YOLO model.
        save_path: Optional path to save the annotated image.

    Returns:
        Annotated image as a NumPy array.
    """
    results = yolo_face_model(img_array)
    annotated_img = results[0].plot()  # Get annotated image from the first result
    if save_path:
        cv2.imwrite(save_path, annotated_img)
    return annotated_img
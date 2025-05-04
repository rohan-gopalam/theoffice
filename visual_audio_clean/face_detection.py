# face_detection.py

# skin and edges filtering are strong, everything turned off rn

import cv2
import numpy as np
from ultralytics import YOLO
import config
import profile_manager  # Import the profile manager module
from scipy.spatial.distance import cosine

class FaceDetector:
    """Basic face detection using YOLO."""
    
    def __init__(self, model_path=config.YOLO_MODEL_PATH, device=config.device, 
                 conf_threshold=config.YOLO_CONF_THRESHOLD):
        """Initialize the face detector with a YOLO model."""
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = self.load_model()
    
    def load_model(self):
        """Loads the YOLO model."""
        print(f"Loading YOLO model from {self.model_path} on {self.device}...")
        try:
            model = YOLO(self.model_path)
            model.to(self.device)
            return model
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise  # Re-raise the exception to halt execution if loading fails
    
    def detect_faces(self, img_array):
        """
        Detect faces in an image using YOLO.
        
        Args:
            img_array: Input image as a NumPy array.
            
        Returns:
            List of bounding boxes [x1, y1, x2, y2] in pixel coordinates.
        """
        results = self.model(img_array)
        face_boxes = []
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # [N,4]
            confs = results[0].boxes.conf.cpu().numpy()  # [N]
            
            # Get class IDs if available (for models with multiple classes)
            if hasattr(results[0].boxes, 'cls'):
                classes = results[0].boxes.cls.cpu().numpy()
                # Filter for the face/person class (depends on your model)
                person_class = 0  # Usually class 0 for person in COCO dataset
                
                for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                    if conf >= self.conf_threshold and cls == person_class:
                        bbox = [int(x) for x in box]
                        if self._validate_face_box(img_array, bbox):
                            face_boxes.append(bbox)
            else:
                # If no class info (face-specific model), still validate
                for box, conf in zip(boxes, confs):
                    if conf >= self.conf_threshold:
                        bbox = [int(x) for x in box]
                        if self._validate_face_box(img_array, bbox):
                            face_boxes.append(bbox)
        
        return face_boxes
    
    def _validate_face_box(self, img_array, bbox):
        return True
        """
        Validate that a bounding box likely contains a face.
        
        Args:
            img_array: Input image as a NumPy array.
            bbox: Bounding box [x1, y1, x2, y2].
            
        Returns:
            bool: True if likely a face, False otherwise.
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are valid
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
            return False
            
        # Check if box is fully within image bounds
        h, w = img_array.shape[:2]
        if x2 > w or y2 > h:
            return False
            
        # Check aspect ratio (faces are roughly square-ish, not extremely elongated)
        width, height = x2 - x1, y2 - y1
        aspect_ratio = width / height
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            return False
            
        # Extract face region
        face_roi = img_array[y1:y2, x1:x2]
        if face_roi.size == 0:
            return False
            
        # Basic skin tone detection (optional)
        # return self._has_skin_tones(face_roi)
    
    def _has_skin_tones(self, face_roi):
        return True
        # """
        # Check if the ROI contains enough skin-like pixels to be a face.
        
        # Args:
        #     face_roi: Face region as a NumPy array.
            
        # Returns:
        #     bool: True if contains sufficient skin tones, False otherwise.
        # """
        # try:
        #     # Convert to HSV color space for better skin tone detection
        #     face_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            
        #     # Define common skin tone range in HSV
        #     # This is a basic range that works for many skin tones
        #     lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        #     upper_skin = np.array([25, 255, 255], dtype=np.uint8)
            
        #     # Create skin mask
        #     skin_mask = cv2.inRange(face_hsv, lower_skin, upper_skin)
            
        #     # Calculate percentage of skin pixels
        #     skin_percentage = np.sum(skin_mask > 0) / (face_roi.shape[0] * face_roi.shape[1])
            
        #     # Face should have a reasonable amount of skin tone pixels (adjust threshold as needed)
        #     return skin_percentage > 0.25
        # except Exception as e:
        #     print(f"Error in skin tone detection: {e}")
        #     return True  # Default to True in case of error to avoid filtering too aggressively
    
    def visualize_detections(self, img_array, save_path=None):
        """
        Use the YOLO model's built-in plotting method to create an annotated image.
        
        Args:
            img_array: Input image as a NumPy array.
            save_path: Optional path to save the annotated image.
            
        Returns:
            Annotated image as a NumPy array.
        """
        results = self.model(img_array)
        annotated_img = results[0].plot()  # Get annotated image from the first result
        
        if save_path:
            cv2.imwrite(save_path, annotated_img)
            
        return annotated_img


class FaceTracker:
    """Advanced face tracking with consistent IDs across frames."""
    
    def __init__(self, model_path=config.YOLO_MODEL_PATH, device=config.device,
                 conf_threshold=config.YOLO_CONF_THRESHOLD, 
                 tracker_config='bytetrack.yaml',
                 embedding_threshold=config.EMBEDDING_THRESHOLD):
        """Initialize the face tracker."""
        self.model_path = model_path
        self.device = device
        # Use a higher confidence threshold to reduce false positives
        self.conf_threshold = conf_threshold  # Minimum 65% confidence
        self.tracker_config = tracker_config
        # Optionally use a stricter threshold for better differentiation
        self.embedding_threshold = embedding_threshold * 0.8 if embedding_threshold else 0.4
        
        # Initialize the model with tracking capabilities
        self.model = self.load_model()
        
        # Dictionary to store tracking profiles
        self.profiles = {}  # {profile_id: {"embedding": emb, "frames_seen": [idx], ...}}
        
        # Dictionary to map YOLO track IDs to our profile IDs
        self.track_id_to_profile_id = {}  # {yolo_track_id: profile_id}
        
        # Store last known crops for each profile
        self.last_known_crops = {}  # {profile_id: cropped_face_image}
        
        # Store detection confidence history for each profile
        self.profile_confidence = {}  # {profile_id: [confidence scores]}
        
        # Keep track of the last frame processed
        self.last_frame_idx = -1
        
        # Define minimum face size relative to frame
        self.min_face_size = 0  # At least 5% of frame width/height
        self.max_face_size = 1   # At most 80% of frame width/height
        
        print(f"FaceTracker initialized with confidence threshold: {self.conf_threshold}")
        print(f"FaceTracker initialized with embedding threshold: {self.embedding_threshold}")
    
    def load_model(self):
        """Loads the YOLO tracking model."""
        print(f"Loading YOLO tracking model from {self.model_path} on {self.device}...")
        try:
            model = YOLO(self.model_path)
            model.to(self.device)
            return model
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise  # Re-raise the exception to halt execution if loading fails
    
    def track_faces(self, img_array, frame_idx):
        """
        Track faces in the current frame with strict filtering for face validation.
        
        Args:
            img_array: Input image as a NumPy array.
            frame_idx: Current frame index.
            
        Returns:
            List of dictionaries with face information.
        """
        # Get image dimensions for filtering
        results = self.model.track(img_array, persist=True, tracker=self.tracker_config)
        all_boxes = results[0].boxes.xyxy.cpu().numpy()
        all_ids   = results[0].boxes.id   .cpu().numpy() if results[0].boxes.id is not None else []
        print(f"[DEBUG] raw track() → {len(all_boxes)} boxes, {len(all_ids)} track IDs")
        
        img_height, img_width = img_array.shape[:2]
        
        # Run YOLO with tracking
        results = self.model.track(img_array, persist=True, tracker=self.tracker_config, verbose=False)
        
        faces_in_frame = []
        
        # Same-frame uniqueness constraint:
        # Keep track of profile IDs already used in this frame
        profiles_used_in_this_frame = set()
        
        # Check if we have valid detections with tracking IDs
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            
            # Get class IDs if available (for models with multiple classes)
            if hasattr(results[0].boxes, 'cls'):
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                # Filter for person class (usually class 0 in COCO)
                person_class = 0
            else:
                # If no class info (face-specific model), create dummy class array
                classes = np.zeros(len(boxes), dtype=int)
                person_class = 0
            
            # Pre-process to get all valid face boxes
            valid_boxes = []
            for i, (track_id, class_id) in enumerate(zip(track_ids, classes)):
                # Skip if confidence is too low
                if confs[i] < self.conf_threshold:
                    continue
                
                # Skip if not a person/face class (for multi-class models)
                if class_id != person_class:
                    continue
                
                # Get box coordinates
                x1, y1, x2, y2 = [int(x) for x in boxes[i]]
                
                # Skip invalid boxes
                if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                    continue
                
                # Size filtering relative to image dimensions
                width, height = x2 - x1, y2 - y1
                if (width < img_width * self.min_face_size or 
                    height < img_height * self.min_face_size):
                    print("Shape")
                    continue
                
                if (width > img_width * self.max_face_size or 
                    height > img_height * self.max_face_size):
                    print("shape")
                    continue
                
                # Aspect ratio check (faces are roughly square-ish)
                aspect_ratio = width / height
                if aspect_ratio < 0.2 or aspect_ratio > 4.0:
                    print("ratio")
                    continue
                
                # Extract face region for validation
                face_roi = img_array[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue
                
                # Validate that this is a face using skin tones
                if not self._validate_face(face_roi):
                    continue
                
                # If it passes all checks, add to valid boxes
                valid_boxes.append({
                    'idx': i,
                    'track_id': int(track_id),
                    'bbox': (x1, y1, x2, y2),
                    'conf': float(confs[i]),
                    'class_id': int(class_id)
                })
            
            # Sort boxes by confidence (process highest confidence faces first)
            valid_boxes.sort(key=lambda x: x['conf'], reverse=True)
            
            # Process each validated face
            for box_info in valid_boxes:
                i = box_info['idx']
                track_id = box_info['track_id']
                x1, y1, x2, y2 = box_info['bbox']
                
                # Extract face region
                face_roi = img_array[y1:y2, x1:x2]
                
                # Process if face region is valid
                if face_roi.size > 0:
                    # Use the provided profile_manager to get embedding or create a profile
                    if hasattr(self, 'get_face_embedding'):
                        embedding = self.get_face_embedding(face_roi)
                    else:
                        # Placeholder for when face embedding extraction is not implemented
                        embedding = np.random.rand(512)  # Simulated embedding
                    
                    # Check if this YOLO track_id is already mapped to a profile
                    if track_id in self.track_id_to_profile_id:
                        profile_id = self.track_id_to_profile_id[track_id]
                        
                        # Check if this profile ID is already used in this frame
                        if profile_id in profiles_used_in_this_frame:
                            # This is a conflict - create a new profile for this face
                            new_pid = self._create_new_profile(embedding, frame_idx)
                            profile_id = new_pid
                            self.track_id_to_profile_id[track_id] = profile_id
                        else:
                            # Profile exists and hasn't been used in this frame yet
                            self.profiles[profile_id]["frames_seen"].append(frame_idx)
                            
                        # Mark this profile as used in this frame
                        profiles_used_in_this_frame.add(profile_id)
                        self.last_known_crops[profile_id] = face_roi
                        
                        # Update confidence history
                        if profile_id not in self.profile_confidence:
                            self.profile_confidence[profile_id] = []
                        self.profile_confidence[profile_id].append(float(confs[i]))
                    else:
                        # No existing mapping for this track_id, find best matching profile
                        # but respect same-frame uniqueness
                        profile_id = self._find_or_create_profile(embedding, frame_idx, profiles_used_in_this_frame)
                        
                        # Map this YOLO track_id to our profile_id
                        self.track_id_to_profile_id[track_id] = profile_id
                        profiles_used_in_this_frame.add(profile_id)
                        self.last_known_crops[profile_id] = face_roi
                        
                        # Initialize confidence history
                        if profile_id not in self.profile_confidence:
                            self.profile_confidence[profile_id] = []
                        self.profile_confidence[profile_id].append(float(confs[i]))
                    
                    # Add face information to the list
                    faces_in_frame.append({
                        'bbox': (x1, y1, x2, y2),
                        'yolo_track_id': int(track_id),
                        'profile_id': profile_id,
                        'confidence': float(confs[i])
                    })
        
        # Update the last processed frame
        self.last_frame_idx = frame_idx
        
        # Optionally filter out low-confidence profiles periodically
        if frame_idx % 10 == 0:
            self._prune_low_confidence_profiles()
        
        return faces_in_frame
    
    def _validate_face(self, face_roi):
        return True
        """
        Comprehensive face validation based on multiple checks.
        
        Args:
            face_roi: Cropped face region.
            
        Returns:
            bool: True if ROI contains a face, False otherwise.
        """
        try:
            # 1. Skin tone detection
            face_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            
            # Define skin tone range in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([25, 255, 255], dtype=np.uint8)
            
            # Get skin mask
            skin_mask = cv2.inRange(face_hsv, lower_skin, upper_skin)
            
            # Calculate percentage of skin pixels
            skin_percentage = np.sum(skin_mask > 0) / (face_roi.shape[0] * face_roi.shape[1])
            
            # Minimum required skin percentage
            if skin_percentage < 0.01:
                print('im racist')
                return False
            
            # 2. Variance check (faces usually have varied brightness)
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            var = np.var(gray)
            
            # Low variance usually means uniform region, not a face
            if var < 100:  # Adjust this threshold based on your needs
                print('im blue')
                return False
            
            # 3. Edge detection (faces have distinctive edges)
            edges = cv2.Canny(gray, 100, 200)
            edge_ratio = np.count_nonzero(edges) / (gray.shape[0] * gray.shape[1])
            
            # Faces should have a reasonable amount of edges
            if edge_ratio < 0.01 or edge_ratio > 0.99:
                print('im edgy')
                return False
            
            # All checks passed
            return True
            
        except Exception as e:
            print(f"Error in face validation: {e}")
            return False  # Fail safe on error
    
    def _find_or_create_profile(self, embedding, frame_idx, profiles_used_in_this_frame):
        """
        Find the best matching profile or create a new one, ensuring same-frame uniqueness.
        
        Args:
            embedding: Face embedding vector.
            frame_idx: Current frame index.
            profiles_used_in_this_frame: Set of profile IDs already used in this frame.
            
        Returns:
            profile_id: The assigned profile ID.
        """
        matched_pid = None
        min_dist = float('inf')
        
        # Find the best match among existing profiles
        for pid, profile_data in self.profiles.items():
            # Skip profiles already used in this frame (same-frame constraint)
            if pid in profiles_used_in_this_frame:
                continue
                
            # Check if embeddings are valid before calculating distance
            if profile_data.get("embedding") is not None and embedding is not None:
                try:
                    dist = self._calculate_distance(profile_data["embedding"], embedding)
                    if dist < self.embedding_threshold and dist < min_dist:
                        min_dist = dist
                        matched_pid = pid
                except Exception as e:
                    print(f"Error comparing embedding for profile {pid}: {e}")
                    continue
        
        if matched_pid is not None:
            # Found a match, update the profile's frame list
            self.profiles[matched_pid]["frames_seen"].append(frame_idx)
            return matched_pid
        else:
            # No match found, create a new profile
            return self._create_new_profile(embedding, frame_idx)
    
    def _create_new_profile(self, embedding, frame_idx):
        """Create a new profile with a unique ID."""
        new_pid = max(self.profiles.keys()) + 1 if self.profiles else 1
        self.profiles[new_pid] = {
            "embedding": embedding,
            "frames_seen": [frame_idx],
            "name": f"Person {new_pid}"  # Initial generic name
        }
        return new_pid
    
    def _calculate_distance(self, emb1, emb2):
        """Calculate distance between embeddings (cosine distance)."""
        return cosine(emb1, emb2)
    
    def _prune_low_confidence_profiles(self, min_confidence=0.6, min_appearances=2):
        """
        Remove profiles that have consistently low confidence scores.
        
        Args:
            min_confidence: Minimum average confidence threshold.
            min_appearances: Minimum number of frame appearances required.
        """
        profiles_to_remove = []
        
        for profile_id, confidences in self.profile_confidence.items():
            if profile_id not in self.profiles:
                continue
                
            # Skip profiles with too few detections
            if len(confidences) < min_appearances:
                continue
                
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences)
            
            # Mark for removal if consistently low confidence
            if avg_confidence < min_confidence:
                profiles_to_remove.append(profile_id)
        
        # Remove marked profiles
        for pid in profiles_to_remove:
            if pid in self.profiles:
                del self.profiles[pid]
            if pid in self.last_known_crops:
                del self.last_known_crops[pid]
            if pid in self.profile_confidence:
                del self.profile_confidence[pid]
                
            # Also update track_id mapping
            track_ids_to_remove = []
            for track_id, profile_id in self.track_id_to_profile_id.items():
                if profile_id == pid:
                    track_ids_to_remove.append(track_id)
                    
            for track_id in track_ids_to_remove:
                del self.track_id_to_profile_id[track_id]
    
    def recluster_profiles(self, eps=config.DBSCAN_EPS):
        """
        Recluster all profiles to consolidate identities.
        Should be called periodically, not every frame.
        """
        # Filter out low-quality profiles before reclustering
        self._prune_low_confidence_profiles()
        
        # Use the profile_manager's reclustering function
        new_assignments, new_profiles = profile_manager.recluster_profiles(self.profiles, eps)
        
        # Update our profiles based on the clustering results
        self.profiles = new_profiles
        
        # We need to update our track_id_to_profile_id mapping
        updated_track_id_map = {}
        for track_id, old_profile_id in self.track_id_to_profile_id.items():
            if old_profile_id in new_assignments:
                updated_track_id_map[track_id] = new_assignments[old_profile_id]
        
        self.track_id_to_profile_id = updated_track_id_map
        
        # Also update last_known_crops
        updated_crops = {}
        for old_pid, new_pid in new_assignments.items():
            if old_pid in self.last_known_crops:
                updated_crops[new_pid] = self.last_known_crops[old_pid]
        
        self.last_known_crops = updated_crops
        
        # Update confidence histories
        updated_confidences = {}
        for old_pid, new_pid in new_assignments.items():
            if old_pid in self.profile_confidence:
                if new_pid not in updated_confidences:
                    updated_confidences[new_pid] = []
                updated_confidences[new_pid].extend(self.profile_confidence[old_pid])
        
        self.profile_confidence = updated_confidences
        
        return new_assignments, new_profiles
    
    def filter_stable_profiles(self, min_frame_percentage=0.1):
        """
        Filter out unstable profiles for final results.
        
        Args:
            min_frame_percentage: Minimum percentage of frames a profile should appear in.
            
        Returns:
            dict: Filtered profiles.
        """
        if not self.profiles:
            return {}
            
        # Get the maximum frame index to determine total frames
        total_frames = self.last_frame_idx + 1
        if total_frames == 0:
            return self.profiles
            
        min_frames = max(2, int(total_frames * min_frame_percentage))
        
        # Filter profiles
        stable_profiles = {}
        for pid, profile in self.profiles.items():
            frames_seen = profile.get("frames_seen", [])
            if len(frames_seen) >= min_frames:
                stable_profiles[pid] = profile
                
        return stable_profiles
    
    def visualize_tracked_faces(self, img_array, faces_in_frame):
        """
        Draw bounding boxes and labels for tracked faces.
        
        Args:
            img_array: Input image as a NumPy array.
            faces_in_frame: List of face dictionaries returned by track_faces.
            
        Returns:
            Annotated image.
        """
        vis_img = img_array.copy()
        
        # Create distinct colors for each profile ID
        def get_color(profile_id):
            # Generate a fixed color based on profile ID
            colors = [
                (0, 255, 0),    # Green
                (255, 0, 0),    # Blue (BGR format)
                (0, 0, 255),    # Red
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
                (128, 0, 255),  # Purple
                (0, 128, 255),  # Orange
                (255, 128, 0),  # Light blue
                (128, 255, 0),  # Light green
            ]
            return colors[profile_id % len(colors)]
        
        for face in faces_in_frame:
            x1, y1, x2, y2 = face['bbox']
            profile_id = face['profile_id']
            yolo_id = face['yolo_track_id']
            
            # Get color based on profile ID for visual distinction
            box_color = get_color(profile_id)
            
            # Draw the bounding box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw the profile ID and YOLO track ID
            label = f"P:{profile_id} Y:{yolo_id}"
            
            # If we have a name in the profile, use it
            if profile_id in self.profiles and "name" in self.profiles[profile_id]:
                name = self.profiles[profile_id]["name"]
                label = f"{name} ({label})"
            
            # Add label with background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis_img, (x1, y1-th-5), (x1+tw+5, y1), (0, 0, 0), -1)
            cv2.putText(vis_img, label, (x1+3, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        return vis_img
    
    def draw_thumbnails(self, canvas, start_y=0, canvas_w=None, thumb_size=64, padding=10):
        """
        Draw thumbnails of tracked faces on a canvas.
        
        Args:
            canvas: Canvas image to draw on.
            start_y: Y-coordinate to start drawing thumbnails.
            canvas_w: Width of the canvas (if None, use canvas width).
            thumb_size: Size of thumbnails.
            padding: Padding between thumbnails.
            
        Returns:
            Updated canvas.
        """
        if canvas_w is None:
            canvas_w = canvas.shape[1]
        
        thumb_x = padding
        thumb_y = start_y + padding
        max_thumbs_per_row = max(1, (canvas_w - 2 * padding) // (thumb_size + padding))
        count = 0
        
        # Get stable profiles only
        stable_profiles = self.filter_stable_profiles()
        sorted_profile_ids = sorted(stable_profiles.keys())
        
        # Function to get color based on profile ID (same as in visualize_tracked_faces)
        def get_color(profile_id):
            colors = [
                (0, 255, 0),    # Green
                (255, 0, 0),    # Blue (BGR format)
                (0, 0, 255),    # Red
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
                (128, 0, 255),  # Purple
                (0, 128, 255),  # Orange
                (255, 128, 0),  # Light blue
                (128, 255, 0),  # Light green
            ]
            return colors[profile_id % len(colors)]
        
        for profile_id in sorted_profile_ids:
            if profile_id not in self.last_known_crops:
                continue
                
            if count >= max_thumbs_per_row:
                thumb_y += thumb_size + padding + 15  # Extra space for label
                thumb_x = padding
                count = 0
            
            crop = self.last_known_crops.get(profile_id)
            if crop is not None and crop.size > 0:
                try:
                    thumb = cv2.resize(crop, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
                    thumb_y1, thumb_y2 = thumb_y, thumb_y + thumb_size
                    thumb_x1, thumb_x2 = thumb_x, thumb_x + thumb_size
                    
                    if thumb_y2 < canvas.shape[0]:
                        # Draw border with color matching the profile ID
                        border_color = get_color(profile_id)
                        cv2.rectangle(canvas, (thumb_x1-2, thumb_y1-2), (thumb_x2+2, thumb_y2+2), border_color, 2)
                        
                        # Place thumbnail
                        canvas[thumb_y1:thumb_y2, thumb_x1:thumb_x2] = thumb
                        
                        # Add label
                        name = self.profiles[profile_id].get("name", f"ID: {profile_id}")
                        # Show confidence if available
                        confidence_str = ""
                        if profile_id in self.profile_confidence and self.profile_confidence[profile_id]:
                            avg_conf = sum(self.profile_confidence[profile_id]) / len(self.profile_confidence[profile_id])
                            confidence_str = f" ({avg_conf:.2f})"
                            
                        label = name + confidence_str
                        
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        label_x = thumb_x1 + max(0, (thumb_size - tw) // 2)
                        label_y = thumb_y2 + 12
                        
                        if label_y + th < canvas.shape[0]:
                            cv2.putText(canvas, label, (label_x, label_y), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
                        
                        thumb_x += thumb_size + padding
                        count += 1
                    else:
                        break  # Out of canvas space
                except Exception as e:
                    print(f"Error drawing thumbnail for profile {profile_id}: {e}")
                    thumb_x += thumb_size + padding
                    count += 1
            else:
                thumb_x += thumb_size + padding
                count += 1
        
        return canvas


# For backward compatibility with the original code
def load_yolo_model(model_path=config.YOLO_MODEL_PATH, device=config.device):
    """Loads the YOLO model (legacy function)."""
    detector = FaceDetector(model_path, device)
    return detector.model

def detect_faces_yolo(img_array, yolo_face_model, conf_threshold=config.YOLO_CONF_THRESHOLD):
    """
    Detect faces in an image using YOLO (legacy function).
    
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
        confs = results[0].boxes.conf.cpu().numpy()  # [N]
        
        # Get class IDs if available
        if hasattr(results[0].boxes, 'cls'):
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            # Filter for person class (usually class 0 in COCO)
            person_class = 0
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                if conf >= conf_threshold and cls == person_class:
                    bbox = [int(x) for x in box]
                    # Additional validation
                    x1, y1, x2, y2 = bbox
                    if x1 < x2 and y1 < y2:
                        width, height = x2 - x1, y2 - y1
                        aspect_ratio = width / height
                        if 0.2 <= aspect_ratio <= 5.0:
                            face_boxes.append(bbox)
        else:
            # No class info (face-specific model)
            for box, conf in zip(boxes, confs):
                if conf >= conf_threshold:
                    bbox = [int(x) for x in box]
                    # Additional validation
                    x1, y1, x2, y2 = bbox
                    if x1 < x2 and y1 < y2:
                        width, height = x2 - x1, y2 - y1
                        aspect_ratio = width / height
                        if 0.2 <= aspect_ratio <= 5.0:
                            face_boxes.append(bbox)
    
    return face_boxes

def visualize_yolo_detections(img_array, yolo_face_model, save_path=None):
    """
    Use the YOLO model's built-in plotting method to create an annotated image (legacy function).
    
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
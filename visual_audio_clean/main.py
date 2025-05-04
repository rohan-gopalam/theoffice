import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
import json
import gc  # For manual garbage collection
import openai  # For OpenAI API calls
import subprocess  # For video creation

# Import configurations and functions from other modules
from fix_cuda_issues import *
import config
from video_processor import split_video_into_frames

# Import the new FaceTracker class instead of the individual functions
from face_detection import FaceTracker, load_yolo_model, detect_faces_yolo, visualize_yolo_detections

from face_analysis import get_face_embedding, analyze_emotions, match_known_face, load_known_faces
from gaze import load_gaze_model, run_gaze_estimation
from profile_manager import update_profiles, recluster_profiles
from transcription import LOCAL_VIDEO_PATH, transcribe_audio_stream, group_transcripts_by_time
from visualization import visualize_all
from llm_output import create_llm_input, create_llm_input_with_bbox
from utils import NumpyEncoder

# Set up OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY", "")  # Make sure to set your API key

globalfps = 30

# --- Helper Function for Processing a Single Face ---
def _process_single_face(face_roi_cv, box_pixels, frame_idx, known_faces, raw_profiles):
    """Processes one detected face: embedding, emotion, profile update."""
    x1, y1, x2, y2 = box_pixels

    # 1. Get Embedding
    embedding = get_face_embedding(face_roi_cv)
    if embedding is None:
        return None, raw_profiles # Cannot proceed without embedding

    # 2. Analyze Emotion
    emotion = analyze_emotions(face_roi_cv)

    # 3. Match to Known Faces (Optional)
    known_identity, similarity_score = None, float('inf')
    if known_faces:
        known_identity, similarity_score = match_known_face(embedding, known_faces)
        # Optional: Apply threshold if match score is too high (cosine distance)
        # if similarity_score > config.KNOWN_FACE_THRESHOLD: known_identity = None

    # 4. Update/Assign Profile ID (Temporal Tracking)
    # Pass a copy or handle immutability if necessary, but direct update is often fine here
    profile_id, raw_profiles = update_profiles(embedding, raw_profiles, frame_idx)

    # If a known identity was found, update the profile's name if it's generic
    if known_identity and raw_profiles[profile_id]["name"].startswith("Person "):
         raw_profiles[profile_id]["name"] = known_identity

    # 5. Store initial face data
    face_data = {
        "frame_idx": frame_idx,
        "bbox_pixels": box_pixels,
        "emotion": emotion,
        "temp_profile_id": profile_id,
        "profile_name": raw_profiles[profile_id]["name"], # Use current name
        "known_identity_match": known_identity,
        "known_identity_score": similarity_score if known_identity else None,
        "gaze": {} # Placeholder
    }

    return face_data, raw_profiles


# --- Updated function that uses the Face Tracker ---
def _process_single_frame_with_tracking(img_path, frame_idx, face_tracker, known_faces, raw_profiles, frame_dimensions):
    """Loads image, tracks faces, processes each face, returns frame data using the FaceTracker."""
    frame_faces_details = []
    frame_norm_boxes = []
    image_tensor = None
    width, height = 0, 0 # Initialize dimensions

    try:
        # Load image and get dimensions
        if img_path not in frame_dimensions:
             with Image.open(img_path) as pil_img_check:
                  frame_dimensions[img_path] = pil_img_check.size
        width, height = frame_dimensions[img_path]

        pil_image = Image.open(img_path).convert("RGB")
        np_image = np.array(pil_image)
        cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        # Use the Face Tracker to process this frame
        tracked_faces = face_tracker.track_faces(cv_image, frame_idx)
        
        print(f"  Frame {frame_idx+1}: Tracked faces = {len(tracked_faces)}")

        # Process each tracked face
        if tracked_faces:
            # Prepare image tensor for gaze once per frame
            image_tensor = config.gaze_transform(pil_image).unsqueeze(0)

            for face_data in tracked_faces:
                x1, y1, x2, y2 = face_data['bbox']
                # Get the face region
                face_roi_cv = cv_image[y1:y2, x1:x2]
                if face_roi_cv.size == 0: 
                    continue

                # Get face embedding (only if not already done by tracker)
                # Note: The tracker has profile management built in, but here we use the existing system
                embedding = get_face_embedding(face_roi_cv)
                if embedding is None:
                    continue

                # Analyze emotion
                emotion = analyze_emotions(face_roi_cv)

                # Match to known faces if provided
                known_identity, similarity_score = None, float('inf')
                if known_faces:
                    known_identity, similarity_score = match_known_face(embedding, known_faces)

                # Use the face tracker's profile_id as the temp_profile_id
                # This leverages YOLO's tracking for more consistent IDs
                profile_id = face_data['profile_id']
                
                # We also update our raw_profiles using the existing system
                # This allows us to continue using the recluster_profiles function
                if profile_id not in raw_profiles:
                    raw_profiles[profile_id] = {
                        "embedding": embedding,
                        "frames_seen": [frame_idx],
                        "name": f"Person {profile_id}"
                    }
                else:
                    raw_profiles[profile_id]["frames_seen"].append(frame_idx)

                # If a known identity was found, update the profile's name
                if known_identity and raw_profiles[profile_id]["name"].startswith("Person "):
                    raw_profiles[profile_id]["name"] = known_identity

                # Create face data for this detection
                processed_face_data = {
                    "frame_idx": frame_idx,
                    "bbox_pixels": (x1, y1, x2, y2),
                    "emotion": emotion,
                    "temp_profile_id": profile_id,
                    "profile_name": raw_profiles[profile_id]["name"],
                    "known_identity_match": known_identity,
                    "known_identity_score": similarity_score if known_identity else None,
                    "gaze": {},  # Placeholder for gaze data
                    "yolo_track_id": face_data['yolo_track_id']  # Store the YOLO track ID for reference
                }

                frame_faces_details.append(processed_face_data)
                
                # Store normalized bbox for gaze model input
                norm_box = [x1/width, y1/height, x2/width, y2/height]
                frame_norm_boxes.append(norm_box)

        return frame_faces_details, frame_norm_boxes, image_tensor, raw_profiles, True

    except FileNotFoundError:
        print(f"\nError: Image file not found at {img_path}. Skipping frame.")
        return [], [], None, raw_profiles, False # Indicate failure
    except Exception as e:
        # Add more specific error info if possible
        import traceback
        print(f"\nError processing image {img_path}: {e}")
        # print(traceback.format_exc()) # Uncomment for full traceback if needed
        return [], [], None, raw_profiles, False


# --- Helper Function to Integrate Gaze, Finalize, and Visualize ---
def _integrate_gaze_and_finalize(image_paths, frame_face_details, gaze_results, all_image_tensors, profile_assignment_map, final_profiles, frame_dimensions):
    """Integrates gaze, assigns final IDs, creates visualizations."""
    final_frame_results = {} # Will contain the {img_path: {"faces": [...]}} structure
    visualizations = {}
    gaze_frame_counter = 0 # Index into gaze_results

    print("\n--- Aggregating Final Results and Visualizations ---")
    # Filter out profiles that appear in too few frames (likely false detections)
    total_frames = len(image_paths)
    min_frame_threshold = max(2, int(total_frames * 0.1))  # At least 10% of frames or 2 frames
    
    stable_profiles = {}
    for profile_id, profile_data in final_profiles.items():
        frames_seen = profile_data.get("frames_seen", [])
        if len(frames_seen) >= min_frame_threshold:
            stable_profiles[profile_id] = profile_data
    
    # Use the filtered profiles
    final_profiles = stable_profiles

    for frame_idx, img_path in enumerate(image_paths):
        faces_in_this_frame = frame_face_details[frame_idx]
        if not faces_in_this_frame: # Skip frames where no faces were initially processed
             final_frame_results[img_path] = {"faces": []}
             continue

        frame_faces_output = [] # Final face data list for this frame
        vis_norm_boxes, vis_names, vis_emotions, vis_profile_ids, vis_inout_scores, vis_heatmaps = [], [], [], [], [], []

        # Determine if gaze data should exist for this frame
        frame_had_faces_for_gaze = all_image_tensors[frame_idx] is not None and any(fd is not None for fd in faces_in_this_frame) # Ensure faces were actually processed

        # Prepare gaze data pointers for this frame if applicable
        frame_gaze_heatmaps = None
        frame_gaze_inout = None
        if gaze_results and frame_had_faces_for_gaze:
             # Check bounds before accessing gaze_results
             if gaze_results.get("heatmap") and gaze_frame_counter < len(gaze_results["heatmap"]):
                 frame_gaze_heatmaps = gaze_results["heatmap"][gaze_frame_counter]
             if gaze_results.get("inout") and gaze_frame_counter < len(gaze_results["inout"]):
                  frame_gaze_inout = gaze_results["inout"][gaze_frame_counter]
             gaze_frame_counter += 1 # Increment counter associated with gaze_results index

        # Process each face detail stored earlier for this frame
        for face_idx, face_data in enumerate(faces_in_this_frame):
            if face_data is None: continue # Skip if face processing failed earlier

            # --- Assign Final Profile ID and Name ---
            temp_id = face_data["temp_profile_id"]
            final_id = profile_assignment_map.get(temp_id, temp_id)  # Use temp_id as fallback
            final_name = final_profiles.get(final_id, {}).get("name", f"Person {final_id}" if final_id else "Unknown")

            face_data["profile_id"] = final_id
            face_data["name"] = final_name

            # --- Integrate Gaze Data ---
            if frame_gaze_heatmaps is not None and face_idx < len(frame_gaze_heatmaps):
                heatmap_data = frame_gaze_heatmaps[face_idx]
                face_data["gaze"]["heatmap"] = heatmap_data.cpu().numpy() if isinstance(heatmap_data, torch.Tensor) else np.array(heatmap_data)

            if frame_gaze_inout is not None and face_idx < len(frame_gaze_inout):
                inout_data = frame_gaze_inout[face_idx]
                inout_score = inout_data.item() if isinstance(inout_data, torch.Tensor) else float(inout_data)
                face_data["gaze"]["inout_score"] = inout_score
                face_data["gaze"]["looking_at_camera"] = inout_score > config.GAZE_INOUT_THRESHOLD

            # --- Clean up temporary data ---
            face_data.pop("temp_profile_id", None)

            frame_faces_output.append(face_data)

            # --- Prepare data for visualization ---
            x1, y1, x2, y2 = face_data["bbox_pixels"]
            width, height = frame_dimensions.get(img_path, (0, 0))
            if width > 0 and height > 0:
                vis_norm_boxes.append([x1/width, y1/height, x2/width, y2/height])
            else:
                vis_norm_boxes.append([0,0,0,0]) # Placeholder

            vis_names.append(face_data["name"])
            vis_emotions.append(face_data["emotion"])
            vis_profile_ids.append(face_data["profile_id"])
            vis_inout_scores.append(face_data["gaze"].get("inout_score"))
            if "heatmap" in face_data["gaze"]:
                vis_heatmaps.append(face_data["gaze"]["heatmap"])

        final_frame_results[img_path] = {"faces": frame_faces_output}

        # --- Generate and Save Visualization ---
        if vis_norm_boxes: # Only visualize if there were faces
            try:
                pil_image = Image.open(img_path).convert("RGB")
                vis_img = visualize_all(
                    pil_image,
                    vis_heatmaps or None, vis_norm_boxes,
                    vis_inout_scores if any(s is not None for s in vis_inout_scores) else None,
                    vis_emotions, vis_names, vis_profile_ids
                )
                visualizations[img_path] = vis_img

                if config.VISUALIZATION_OUTPUT_DIR:
                    os.makedirs(config.VISUALIZATION_OUTPUT_DIR, exist_ok=True)
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    out_path = os.path.join(config.VISUALIZATION_OUTPUT_DIR, f"viz_{base_name}.png")
                    vis_img.save(out_path)
            except Exception as e:
                print(f"\nError generating/saving visualization for {img_path}: {e}")

    return final_frame_results, visualizations


# --- Process entire video at once (no chunking) ---
def process_entire_video(video_path):
    """
    Process a video in its entirety, without chunking.
    Uses the FaceTracker for more consistent tracking across frames.
    
    Args:
        video_path: Path to the video file
    """
    print(f"--- Starting Video Analysis Pipeline with Face Tracking ---")
    print(f"Processing video: {video_path}")
    
    # Get video metadata
    import cv2
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"FATAL ERROR: Could not open video file at '{video_path}'")
        return
    
    fps = video.get(cv2.CAP_PROP_FPS)
    global globalfps
    globalfps = fps
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = frame_count / fps
    
    video.release()
    
    print(f"Video stats: {frame_count} frames, {fps:.2f} FPS, {total_duration:.2f} seconds")
    
    # Load models once
    try:
        print("\nLoading models...")
        # Initialize the Face Tracker
        face_tracker = FaceTracker(model_path=config.YOLO_MODEL_PATH, 
                                 device=config.device,
                                 conf_threshold=config.YOLO_CONF_THRESHOLD)
        
        gaze_model = load_gaze_model()
        
        # Optional: Load known faces
        known_face_images = None  # Set this if you have known faces
        known_faces_embeddings = None
        if known_face_images and face_tracker.model:
            print("Loading known face embeddings...")
            known_faces_embeddings = load_known_faces(known_face_images, face_tracker.model)
    except Exception as e:
        print(f"\nFATAL ERROR: Failed to load models. Exiting. Error: {e}")
        return
    
    # Create output directories
    os.makedirs(config.FRAME_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(config.VISUALIZATION_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.LLM_OUTPUT_PATH), exist_ok=True)
    
    # 1. Extract frames from the video
    start_time = time.time()
    print(f"Extracting frames...")
    
    image_paths = split_video_into_frames(
        video_path,
        config.FRAME_OUTPUT_FOLDER,
        frames_per_second=config.FRAMES_PER_SECOND_TO_EXTRACT
    )
    
    if not image_paths:
        print(f"No frames extracted. Exiting.")
        return
    
    print(f"Extracted {len(image_paths)} frames")
    
    # 2. Run full analysis on the entire video
    try:
        print(f"Running analysis...")
        analysis_results, llm_json_data = analyze_video_frames_with_tracking(
            image_paths,
            0,  # No chunk index since we're processing the entire video
            face_tracker,
            gaze_model,
            known_faces=known_faces_embeddings
        )
        
        # 3. Display/save results
        display_analysis_results(analysis_results)
        
        # 4. Add metadata to the JSON
        llm_json_data["video_metadata"] = {
            "total_frames": len(image_paths),
            "fps": fps,
            "duration": total_duration
        }
        
        # 5. Save the JSON
        standard_json_path = config.LLM_OUTPUT_PATH
        with_bbox_json_path = os.path.join(
            os.path.dirname(config.LLM_OUTPUT_PATH),
            f"{os.path.splitext(os.path.basename(config.LLM_OUTPUT_PATH))[0]}_with_bbox.json"
        )
        
        # Save the standard JSON (should already be done by create_llm_input, but just to be sure)
        with open(standard_json_path, 'w') as f:
            json.dump(llm_json_data, f, indent=2, cls=NumpyEncoder)
        
        # Create and save the JSON with bounding boxes
        llm_json_with_bbox = create_llm_input_with_bbox(analysis_results, transcribe_audio_stream(config.VIDEO_INPUT_PATH), with_bbox_json_path, globalfps)
        
        print(f"Analysis completed and saved to:")
        print(f"  - Standard JSON: {standard_json_path}")
        print(f"  - JSON with bounding boxes: {with_bbox_json_path}")
        
        # 6. AI Video Editing - Step 1: Identify main characters and framing preferences
        main_character_analysis = analyze_main_characters(standard_json_path)
        with open(os.path.join(config.OUTPUT_DIR, "main_character_analysis.json"), 'w') as f:
            json.dump(main_character_analysis, f, indent=2)
        
        # 7. AI Video Editing - Step 2: Generate frame-by-frame crop recommendations
        crop_recommendations = generate_crop_recommendations(with_bbox_json_path, main_character_analysis)
        with open(os.path.join(config.OUTPUT_DIR, "crop_recommendations.json"), 'w') as f:
            json.dump(crop_recommendations, f, indent=2)
        
        # 8. AI Video Editing - Step 3: Create the edited video
        edited_video_path = os.path.join(config.OUTPUT_DIR, f"{base_name}_edited.mp4")
        create_edited_video(video_path, crop_recommendations, edited_video_path, fps)
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
    
    # 9. Clean up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()  # Force garbage collection
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    
    print("\n--- Video Analysis Pipeline Complete ---")
    print(f"Results saved to {config.OUTPUT_DIR}")
    print(f"Edited video saved to {edited_video_path}")


# --- Updated analysis function for entire video ---
def analyze_video_frames_with_tracking(image_paths, chunk_idx, face_tracker, gaze_model, known_faces=None):
    """Orchestrates the frame-by-frame analysis using face tracking for better consistency."""
    start_time = time.time()
    overall_results = {}
    raw_profiles = {} # Tracks faces across frames before reclustering
    all_frame_face_details = [] # List of lists: [[face1_data, face2_data], [face3_data], ...]
    all_norm_boxes = [] # List of lists: [[norm_box1, norm_box2], [norm_box3], ...]
    all_image_tensors = [] # List of tensors or None
    frame_dimensions = {} # Cache frame dimensions

    print(f"\nStarting analysis of {len(image_paths)} frames with face tracking...")
    if not image_paths: return {}, {} # Handle empty input

    # --- Stage 1: Process Each Frame with Face Tracking ---
    for frame_idx, img_path in enumerate(image_paths):
        print(f"\rProcessing Frame {frame_idx+1}/{len(image_paths)}...", end="")
        # Call updated helper to process this frame with tracking
        frame_details, norm_boxes, img_tensor, raw_profiles, success = _process_single_frame_with_tracking(
            img_path, frame_idx, face_tracker, known_faces, raw_profiles, frame_dimensions
        )
        # Store results from this frame
        all_frame_face_details.append(frame_details)
        all_norm_boxes.append(norm_boxes)
        all_image_tensors.append(img_tensor) # Will be None if no faces or error
        # raw_profiles is updated in-place by the helper function

    print("\n--- Stage 2: Batch Gaze Estimation ---")
    valid_tensors = [t for t in all_image_tensors if t is not None]
    # Filter norm_boxes corresponding to valid tensors
    valid_norm_boxes = [boxes for boxes, tensor in zip(all_norm_boxes, all_image_tensors) if tensor is not None]

    gaze_results = None
    if valid_tensors and any(valid_norm_boxes): # Check if there are actually boxes to process
        gaze_results = run_gaze_estimation(gaze_model, valid_tensors, valid_norm_boxes)
    else:
        print("Skipping gaze estimation: No faces detected or tensors generated.")

    # --- Stage 3: Recluster Profiles ---
    print("\n--- Stage 3: Reclustering Profiles ---")
    valid_raw_profiles = {pid: data for pid, data in raw_profiles.items() if data.get("embedding") is not None}
    profile_assignment_map, final_profiles = recluster_profiles(valid_raw_profiles)

    # --- Stage 4: Integrate Gaze, Finalize Results, and Visualize ---
    final_frame_results, visualizations = _integrate_gaze_and_finalize(
        image_paths, all_frame_face_details, gaze_results, all_image_tensors,
        profile_assignment_map, final_profiles, frame_dimensions
    )

    # -- Stage 5: Get full transcript --
    transcript = transcribe_audio_stream(config.VIDEO_INPUT_PATH)

    # Combine results
    overall_results = final_frame_results # Add the frame data
    overall_results["profiles"] = final_profiles
    overall_results["visualizations"] = visualizations

    # --- Stage 5: Generate LLM Input ---
    print("\n--- Stage 5: Generating LLM Input JSON ---")
    llm_data = create_llm_input(overall_results, transcript, config.LLM_OUTPUT_PATH, globalfps)

    end_time = time.time()
    print(f"\nAnalysis complete. Total time: {end_time - start_time:.2f} seconds.")

    return overall_results, llm_data

# --- AI Editing Function: Analyze main characters ---
def analyze_main_characters(json_path):
    """
    Use OpenAI API to identify main characters and framing preferences for each frame.
    
    Args:
        json_path: Path to the standard JSON file (without bounding boxes)
        
    Returns:
        Dictionary with main character analysis
    """
    print("\n--- AI Editing: Analyzing Main Characters ---")
    
    try:
        # Load the JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Prepare the prompt for OpenAI API
        prompt = """
You are an expert film editor. Based on the facial analysis of a video, I need you to identify who is the main focus in each frame.

For each frame, analyze all people present and determine:
1. Who is the main character in this frame
2. Whether we should focus on their face (1), their body (2), or stay zoomed out (-1)
3. Why this framing choice is appropriate based on emotions, gaze direction, and scene context

You must use a consistent format in your response. For each frame, provide:
- Frame identifier
- Main character name
- Framing choice (1=face, 2=body, -1=zoomed out)
- Brief explanation

IMPORTANT: Ensure smooth transitions between frames. The main character shouldn't change too frequently and framing should gradually change, not jump back and forth abruptly.

Here is the complete analysis of the video:
"""
        
        # Add the JSON data to the prompt
        prompt += json.dumps(data, indent=2)
        
        # Call OpenAI API using the updated syntax for openai>=1.0.0
        print("Calling OpenAI API to analyze main characters...")
        
        from openai import OpenAI
        client = OpenAI()
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in video editing."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.3
        )
        
        # Extract the response text - updated syntax
        response_text = response.choices[0].message.content
        
        # Now structure the response into a frame-by-frame format
        # Create a dictionary to store the result
        print("Processing OpenAI's response...")
        
        result = {
            "frame_analysis": [],
            "main_characters": {}
        }
        
        # First identify character mentions and count them
        character_mentions = {}
        
        # Process each frame in the original data
        for frame_entry in data.get("frame_by_frame_analysis", []):
            frame_id = frame_entry.get("frame_identifier", "")
            
            # Parse the response text for this frame
            frame_info = None
            for line in response_text.split('\n'):
                if frame_id in line:
                    # Found reference to this frame
                    parts = line.split('\n')
                    # Extract character name and framing choice
                    for part in parts:
                        if "Main character:" in part:
                            character = part.split("Main character:")[1].strip()
                            if character in character_mentions:
                                character_mentions[character] += 1
                            else:
                                character_mentions[character] = 1
        
        # Identify main characters (those mentioned most frequently)
        sorted_characters = sorted(character_mentions.items(), key=lambda x: x[1], reverse=True)
        main_chars = [char for char, count in sorted_characters[:3]]  # Get top 3
        result["main_characters"] = {char: count for char, count in sorted_characters[:3]}
        
        # Now process each frame again to extract the analysis
        for frame_entry in data.get("frame_by_frame_analysis", []):
            frame_id = frame_entry.get("frame_identifier", "")
            frame_analysis = {
                "frame_id": frame_id,
                "main_character": None,
                "framing_choice": -1,  # Default to zoomed out
                "explanation": ""
            }
            
            # Search for this frame in the response
            frame_section = None
            for section in response_text.split("\n\n"):
                if frame_id in section:
                    frame_section = section
                    break
            
            if frame_section:
                # Extract main character
                if "Main character:" in frame_section:
                    character = frame_section.split("Main character:")[1].split("\n")[0].strip()
                    frame_analysis["main_character"] = character
                
                # Extract framing choice
                if "Framing choice:" in frame_section:
                    framing_text = frame_section.split("Framing choice:")[1].split("\n")[0].strip()
                    if "1" in framing_text:
                        frame_analysis["framing_choice"] = 1
                    elif "2" in framing_text:
                        frame_analysis["framing_choice"] = 2
                    else:
                        frame_analysis["framing_choice"] = -1
                
                # Extract explanation
                if "Explanation:" in frame_section:
                    explanation = frame_section.split("Explanation:")[1].strip()
                    frame_analysis["explanation"] = explanation
            
            result["frame_analysis"].append(frame_analysis)
        
        print(f"Main character analysis complete. Identified {len(main_chars)} main characters.")
        return result
    
    except Exception as e:
        print(f"Error analyzing main characters: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "frame_analysis": [], "main_characters": {}}


# --- AI Editing Function: Generate crop recommendations ---
def generate_crop_recommendations(with_bbox_json_path, main_character_analysis):
    """
    Use OpenAI API to generate frame-by-frame crop recommendations.
    
    Args:
        with_bbox_json_path: Path to the JSON file with bounding boxes
        main_character_analysis: Output from the analyze_main_characters function
        
    Returns:
        Dictionary with crop recommendations for each frame
    """
    print("\n--- AI Editing: Generating Crop Recommendations ---")
    
    try:
        # Load the JSON data with bounding boxes
        with open(with_bbox_json_path, 'r') as f:
            bbox_data = json.load(f)
        
        # Prepare the prompt for OpenAI API
        prompt = """
You are an expert film editor. I need you to create precise frame-by-frame crop recommendations for a video.

I'll provide:
1. JSON data with bounding boxes for all people in each frame
2. A previous analysis that identifies the main character in each frame and suggested framing

Your task is to determine EXACT crop coordinates for each frame that:
1. Focuses on the main character identified in the previous analysis
2. Uses the suggested framing (1=face close-up, 2=upper body, -1=zoomed out)
3. Creates SMOOTH transitions between frames (no jarring jumps)
4. Maintains the same aspect ratio throughout

For each frame, provide:
- Frame identifier
- Crop coordinates as [x1, y1, x2, y2] where:
  * x1, y1 = top-left corner (in pixels)
  * x2, y2 = bottom-right corner (in pixels)
- Ensure the crop maintains the original aspect ratio
- Make transitions between frames gradual (no more than 5% change in coordinates between consecutive frames)

Previous framing analysis:
"""

        # Add the main character analysis to the prompt
        prompt += json.dumps(main_character_analysis, indent=2)
        
        prompt += "\n\nJSON data with bounding boxes:\n"
        
        # Add the bounding box data to the prompt
        prompt += json.dumps(bbox_data, indent=2)
        
        # Call OpenAI API using the updated syntax for openai>=1.0.0
        print("Calling OpenAI API to generate crop recommendations...")
        
        from openai import OpenAI
        client = OpenAI()
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in video editing."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.3
        )
        
        # Extract the response text - updated syntax
        response_text = response.choices[0].message.content
        
        # Parse the crop recommendations
        print("Processing OpenAI's crop recommendations...")
        
        # Create a dictionary to store the crop recommendations
        result = {
            "crop_recommendations": []
        }
        
        # Extract video dimensions from the first frame if available
        video_width = 1920  # Default width
        video_height = 1080  # Default height
        
        # Try to get actual dimensions from the first frame with people
        for frame_entry in bbox_data.get("frame_by_frame_analysis", []):
            people = frame_entry.get("people_in_frame", [])
            if people and "bbox" in people[0]:
                # Estimate video dimensions based on the bounding box
                bbox = people[0]["bbox"]
                # Assume bounding box doesn't take up the entire frame
                # This is an estimate and might need adjustment
                video_width = max(video_width, bbox[2] * 2)
                video_height = max(video_height, bbox[3] * 2)
                break
        
        result["video_dimensions"] = {
            "width": video_width,
            "height": video_height
        }
        
        # Process OpenAI's response
        # Look for frame identifiers and crop coordinates
        import re
        pattern = r"Frame: ([^\n]+)[\s\S]*?Crop coordinates: \[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
        matches = re.findall(pattern, response_text)
        
        # If the regex pattern doesn't find matches, try a different approach
        if not matches:
            print("Using alternative parsing for crop recommendations...")
            # For each frame in the original data, search for its ID in the response
            for frame_entry in bbox_data.get("frame_by_frame_analysis", []):
                frame_id = frame_entry.get("frame_identifier", "")
                
                # Find this frame ID in the response
                frame_section = None
                for section in response_text.split("\n\n"):
                    if frame_id in section:
                        frame_section = section
                        break
                
                if frame_section:
                    # Look for coordinates in this section
                    coord_match = re.search(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", frame_section)
                    if coord_match:
                        x1, y1, x2, y2 = map(int, coord_match.groups())
                        result["crop_recommendations"].append({
                            "frame_id": frame_id,
                            "crop_coordinates": [x1, y1, x2, y2]
                        })
        else:
            # Process the regex matches
            for match in matches:
                frame_id, x1, y1, x2, y2 = match
                result["crop_recommendations"].append({
                    "frame_id": frame_id.strip(),
                    "crop_coordinates": [int(x1), int(y1), int(x2), int(y2)]
                })
        
        # Ensure we have at least some crop recommendations
        if not result["crop_recommendations"]:
            print("Warning: No crop recommendations found in the OpenAI response. Using default crops.")
            # Generate default crop recommendations for each frame
            for frame_entry in bbox_data.get("frame_by_frame_analysis", []):
                frame_id = frame_entry.get("frame_identifier", "")
                # Default to center crop at 80% of original size
                x1 = int(video_width * 0.1)
                y1 = int(video_height * 0.1)
                x2 = int(video_width * 0.9)
                y2 = int(video_height * 0.9)
                
                # If there are people in the frame, try to center on them
                people = frame_entry.get("people_in_frame", [])
                if people:
                    # Find the main character if possible
                    main_char = None
                    for analysis in main_character_analysis.get("frame_analysis", []):
                        if analysis.get("frame_id") == frame_id:
                            main_char = analysis.get("main_character")
                            break
                    
                    # Find the main character in the people list
                    main_person = None
                    for person in people:
                        if main_char and person.get("name") == main_char and "bbox" in person:
                            main_person = person
                            break
                    
                    # If main character found, center crop on them
                    if main_person and "bbox" in main_person:
                        bbox = main_person["bbox"]
                        center_x = (bbox[0] + bbox[2]) // 2
                        center_y = (bbox[1] + bbox[3]) // 2
                        
                        # Adjust crop to center on this person
                        crop_width = int(video_width * 0.8)
                        crop_height = int(video_height * 0.8)
                        
                        x1 = max(0, center_x - crop_width // 2)
                        y1 = max(0, center_y - crop_height // 2)
                        x2 = min(video_width, x1 + crop_width)
                        y2 = min(video_height, y1 + crop_height)
                
                result["crop_recommendations"].append({
                    "frame_id": frame_id,
                    "crop_coordinates": [x1, y1, x2, y2]
                })
        
        # Post-process to ensure smooth transitions
        smooth_crop_recommendations = smooth_transitions(result["crop_recommendations"])
        result["crop_recommendations"] = smooth_crop_recommendations
        
        print(f"Generated {len(result['crop_recommendations'])} crop recommendations.")
        return result
    
    except Exception as e:
        print(f"Error generating crop recommendations: {e}")
        import traceback
        traceback.print_exc()
        # Return an empty but valid structure to prevent KeyError in downstream functions
        return {"error": str(e), "crop_recommendations": [], "video_dimensions": {"width": 1920, "height": 1080}}
# --- Helper function: Smooth transitions between frames ---
def smooth_transitions(crop_recommendations, max_delta_percent=0.05):
    """
    Ensure smooth transitions between frame crops.
    
    Args:
        crop_recommendations: List of crop recommendations
        max_delta_percent: Maximum percentage change between consecutive frames
        
    Returns:
        List of smoothed crop recommendations
    """
    if not crop_recommendations or len(crop_recommendations) <= 1:
        return crop_recommendations
    
    # Sort by frame ID to ensure chronological order
    sorted_crops = sorted(crop_recommendations, key=lambda x: x["frame_id"])
    
    # Initialize smoothed list with the first recommendation
    smoothed = [sorted_crops[0]]
    
    # Process each subsequent recommendation
    for i in range(1, len(sorted_crops)):
        prev_crop = smoothed[-1]["crop_coordinates"]
        curr_crop = sorted_crops[i]["crop_coordinates"]
        
        # Calculate maximum allowed change
        max_delta_x = max(prev_crop[2] - prev_crop[0], 1) * max_delta_percent
        max_delta_y = max(prev_crop[3] - prev_crop[1], 1) * max_delta_percent
        
        # Calculate actual changes
        delta_x1 = curr_crop[0] - prev_crop[0]
        delta_y1 = curr_crop[1] - prev_crop[1]
        delta_x2 = curr_crop[2] - prev_crop[2]
        delta_y2 = curr_crop[3] - prev_crop[3]
        
        # Clamp changes to max allowed
        smooth_x1 = prev_crop[0] + max(-max_delta_x, min(max_delta_x, delta_x1))
        smooth_y1 = prev_crop[1] + max(-max_delta_y, min(max_delta_y, delta_y1))
        smooth_x2 = prev_crop[2] + max(-max_delta_x, min(max_delta_x, delta_x2))
        smooth_y2 = prev_crop[3] + max(-max_delta_y, min(max_delta_y, delta_y2))
        
        # Create smoothed crop
        smoothed.append({
            "frame_id": sorted_crops[i]["frame_id"],
            "crop_coordinates": [int(smooth_x1), int(smooth_y1), int(smooth_x2), int(smooth_y2)]
        })
    
    return smoothed


# --- AI Editing Function: Create edited video ---
def create_edited_video(video_path, crop_recommendations, output_path, fps=30):
    """
    Create an edited video with the recommended crops.
    
    Args:
        video_path: Path to the original video
        crop_recommendations: Crop recommendations dictionary
        output_path: Path to save the edited video
        fps: Frames per second
    """
    print("\n--- AI Editing: Creating Edited Video ---")
    
    try:
        # Create a mapping of frame ID to crop coordinates
        crop_map = {rec["frame_id"]: rec["crop_coordinates"] for rec in crop_recommendations["crop_recommendations"]}
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create a temporary directory for cropped frames
        temp_dir = os.path.join(config.OUTPUT_DIR, "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Process each frame
        frame_idx = 0
        processed_count = 0
        
        print(f"Processing video frames ({frame_count} total)...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Generate frame ID in the same format as used in crop recommendations
            frame_id = f"frame_{frame_idx:04d}.jpg"  # Adjust format as needed
            
            # Check if we have a crop recommendation for this frame
            if frame_id in crop_map:
                x1, y1, x2, y2 = crop_map[frame_id]
                
                # Ensure coordinates are within frame boundaries
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(x1 + 1, min(x2, width))
                y2 = max(y1 + 1, min(y2, height))
                
                # Crop the frame
                cropped_frame = frame[y1:y2, x1:x2]
                
                # Resize to maintain consistent output size
                # You can adjust output_width and output_height as needed
                output_width = 1280
                output_height = 720
                
                # Preserve aspect ratio
                crop_aspect = (x2 - x1) / (y2 - y1)
                output_aspect = output_width / output_height
                
                if crop_aspect > output_aspect:  # Crop is wider than output
                    new_width = output_width
                    new_height = int(output_width / crop_aspect)
                else:  # Crop is taller than output
                    new_height = output_height
                    new_width = int(output_height * crop_aspect)
                
                resized_frame = cv2.resize(cropped_frame, (new_width, new_height))
                
                # Create a blank frame with the output dimensions
                final_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                
                # Center the resized frame on the blank frame
                y_offset = (output_height - new_height) // 2
                x_offset = (output_width - new_width) // 2
                
                final_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
                
                # Save the frame
                output_frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(output_frame_path, final_frame)
                processed_count += 1
            else:
                # If no crop recommendation, use default center crop
                center_x = width // 2
                center_y = height // 2
                
                half_width = min(width, height) // 2
                half_height = half_width
                
                x1 = max(0, center_x - half_width)
                y1 = max(0, center_y - half_height)
                x2 = min(width, center_x + half_width)
                y2 = min(height, center_y + half_height)
                
                # Crop the frame
                cropped_frame = frame[y1:y2, x1:x2]
                
                # Resize to output dimensions
                output_width = 1280
                output_height = 720
                resized_frame = cv2.resize(cropped_frame, (output_width, output_height))
                
                # Save the frame
                output_frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(output_frame_path, resized_frame)
                processed_count += 1
            
            # Print progress
            if frame_idx % 10 == 0:
                print(f"\rProcessed {frame_idx}/{frame_count} frames...", end="")
            
            frame_idx += 1
        
        print(f"\nProcessed {processed_count} frames. Creating video...")
        
        # Release the video capture
        cap.release()
        
        # Use ffmpeg to create the video from the frames
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-framerate", str(fps),
            "-i", os.path.join(temp_dir, "frame_%06d.jpg"),
            "-c:v", "libx264",
            "-profile:v", "high",
            "-crf", "18",  # Quality setting (lower = better quality, 18 is visually lossless)
            "-pix_fmt", "yuv420p",
            output_path
        ]
        
        print(f"Running ffmpeg command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        print(f"Edited video saved to {output_path}")
        
        # Optionally, clean up temporary files
        # import shutil
        # shutil.rmtree(temp_dir)
        
        return output_path
    
    except Exception as e:
        print(f"Error creating edited video: {e}")
        import traceback
        traceback.print_exc()
        return None


# Keep display_analysis_results function unchanged
def display_analysis_results(results):
    """Saves the generated visualizations to files without displaying them."""
    visual_out = results.get("visualizations", {})
    if not visual_out:
        print("\nNo visualizations were generated or found in results.")
    
    print("\nSaving Visualizations to files...")
    for path, vis_img in visual_out.items():
        try:
            # The image should already be saved by _integrate_gaze_and_finalize,
            # but we can add a redundant save here to be sure
            if config.VISUALIZATION_OUTPUT_DIR:
                os.makedirs(config.VISUALIZATION_OUTPUT_DIR, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(path))[0]
                out_path = os.path.join(config.VISUALIZATION_OUTPUT_DIR, f"display_{base_name}.png")
                vis_img.save(out_path)
                print(f"Saved visualization to {out_path}")
        except Exception as e:
            print(f"Error saving visualization for {os.path.basename(path)}: {e}")

    # Print profile summary
    profs = results.get("profiles", {})
    if profs:
        print("\n--- Final Profile Summary ---")
        for pid, prof in sorted(profs.items()):
            frame_indices = prof.get('frames_seen', [])
            appearance_count = len(frame_indices)
            print(f"  Profile ID {pid} ({prof.get('name', 'Unknown')}): Appeared in {appearance_count} frame(s)")
    else:
        print("\nNo distinct profiles were finalized.")

# -----------------------------
# Main Execution Block
# -----------------------------
  
if __name__ == "__main__":
    import sys
    import os
    
    # Handle CLI input
    video_path = sys.argv[1] if len(sys.argv) > 1 else config.VIDEO_INPUT_PATH

    # Dynamically derive output directory name
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    config.OUTPUT_DIR = os.path.join("output", base_name)
    
    # Update all output paths based on the base output directory
    config.FRAME_OUTPUT_FOLDER = os.path.join(config.OUTPUT_DIR, "extracted_frames")
    config.VISUALIZATION_OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "visualizations")
    config.LLM_OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, "llm_analysis_input.json")

    # Also update this for completeness
    config.VIDEO_INPUT_PATH = video_path

    print(f"📽️  Processing {video_path}")
    print(f"📁  Saving outputs to {config.OUTPUT_DIR}")

    ###
    print("--- Starting Video Analysis Pipeline with Face Tracking ---")
    
    if not os.path.exists(config.VIDEO_INPUT_PATH):
        print(f"FATAL ERROR: Input video file not found at '{config.VIDEO_INPUT_PATH}'")
        print("Please check the VIDEO_INPUT_PATH in config.py")
        exit(1)
    
    # Process the video without chunking
    process_entire_video(config.VIDEO_INPUT_PATH)
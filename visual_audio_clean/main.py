# main.py
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
from llm_output import create_llm_input

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


# --- Updated process_video_in_chunks function ---
def process_video_in_chunks(video_path, chunk_duration=30):
    """
    Process a video in fixed-duration chunks, saving results for each chunk separately.
    Uses the FaceTracker for more consistent tracking across frames.
    
    Args:
        video_path: Path to the video file
        chunk_duration: Duration of each chunk in seconds
    """
    print(f"--- Starting Chunked Video Analysis Pipeline with Face Tracking ---")
    print(f"Processing video: {video_path} in {chunk_duration}-second chunks")
    
    # Get video metadata (total duration)
    import cv2
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"FATAL ERROR: Could not open video file at '{video_path}'")
        return
    
    fps = video.get(cv2.CAP_PROP_FPS)
    globalfps = fps
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = frame_count / fps
    total_chunks = int(total_duration / chunk_duration) + 1
    video.release()
    
    print(f"Video stats: {frame_count} frames, {fps:.2f} FPS, {total_duration:.2f} seconds")
    print(f"Will process in {total_chunks} chunks of {chunk_duration} seconds each")
    
    # Load models once (outside the chunk loop)
    try:
        print("\nLoading models...")
        # Initialize the Face Tracker instead of just the YOLO model
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
    
    # Process each chunk
    for chunk_idx in range(total_chunks):
        chunk_start_time = chunk_idx * chunk_duration
        chunk_end_time = min((chunk_idx + 1) * chunk_duration, total_duration)
        
        print(f"\n\n=== Processing Chunk {chunk_idx+1}/{total_chunks} ===")
        print(f"Time range: {chunk_start_time:.2f}s - {chunk_end_time:.2f}s")
        
        # Create a temporary output directory for this chunk
        chunk_output_dir = f"{config.OUTPUT_DIR}/chunk_{chunk_idx+1}"
        chunk_frame_dir = f"{chunk_output_dir}/frames"
        chunk_viz_dir = f"{chunk_output_dir}/visualizations"
        chunk_json_path = f"{chunk_output_dir}/analysis_chunk_{chunk_idx+1}.json"
        
        os.makedirs(chunk_frame_dir, exist_ok=True)
        os.makedirs(chunk_viz_dir, exist_ok=True)
        
        # 1. Extract frames only for this time range
        start_time = time.time()
        print(f"Extracting frames for chunk {chunk_idx+1}...")
        
        # Modify the frame extraction function to accept time range parameters
        chunk_image_paths = split_video_into_frames(
            video_path,
            chunk_frame_dir,
            frames_per_second=config.FRAMES_PER_SECOND_TO_EXTRACT,
            start_time=chunk_start_time,
            end_time=chunk_end_time
        )
        
        if not chunk_image_paths:
            print(f"No frames extracted for chunk {chunk_idx+1}. Skipping.")
            continue
        
        print(f"Extracted {len(chunk_image_paths)} frames for chunk {chunk_idx+1}")
        
        # 2. Temporarily override config for this chunk
        original_viz_dir = config.VISUALIZATION_OUTPUT_DIR
        original_llm_path = config.LLM_OUTPUT_PATH
        
        config.VISUALIZATION_OUTPUT_DIR = chunk_viz_dir
        config.LLM_OUTPUT_PATH = chunk_json_path
        
        # 3. Run full analysis on this chunk
        try:
            print(f"Running analysis on chunk {chunk_idx+1}...")
            analysis_results, llm_json_data = analyze_video_frames_with_tracking(
                chunk_image_paths,
                chunk_idx,
                face_tracker,
                gaze_model,
                known_faces=known_faces_embeddings
            )
            
            # 4. Display/save results for this chunk
            display_analysis_results(analysis_results)
            
            # 5. Add chunk metadata to the JSON
            llm_json_data["chunk_metadata"] = {
                "chunk_index": chunk_idx + 1,
                "total_chunks": total_chunks,
                "start_time": chunk_start_time,
                "end_time": chunk_end_time,
                "frame_count": len(chunk_image_paths)
            }
            
            # 6. Save the JSON explicitly (should already be done by create_llm_input, but just to be sure)
            with open(chunk_json_path, 'w') as f:
                json.dump(llm_json_data, f, indent=2)
                
            print(f"Analysis for chunk {chunk_idx+1} completed and saved to {chunk_json_path}")
            
        except Exception as e:
            print(f"Error processing chunk {chunk_idx+1}: {e}")
            import traceback
            traceback.print_exc()
        
        # 7. Restore original config
        config.VISUALIZATION_OUTPUT_DIR = original_viz_dir
        config.LLM_OUTPUT_PATH = original_llm_path
        
        # 8. Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()  # Force garbage collection
        
        chunk_end_time = time.time()
        print(f"Chunk {chunk_idx+1} processing time: {chunk_end_time - start_time:.2f} seconds")
    
    print("\n--- Chunked Video Analysis Pipeline Complete ---")
    print(f"Results saved to {config.OUTPUT_DIR}/chunk_* directories")


# --- Updated Main Analysis Function (using Face Tracker) ---
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

    # -- Stage 5 --
    transcript = transcribe_audio_stream(config.VIDEO_INPUT_PATH, chunk_size=30)[chunk_idx]

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
    print("--- Starting Chunked Video Analysis Pipeline with Face Tracking ---")
    
    if not os.path.exists(config.VIDEO_INPUT_PATH):
        print(f"FATAL ERROR: Input video file not found at '{config.VIDEO_INPUT_PATH}'")
        print("Please check the VIDEO_INPUT_PATH in config.py")
        exit(1)
    
    # Process the video in chunks using face tracking
    process_video_in_chunks(
        config.VIDEO_INPUT_PATH,
        chunk_duration=30  # 30-second chunks
    )
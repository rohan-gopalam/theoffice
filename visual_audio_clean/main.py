# main.py
import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time

# Import configurations and functions from other modules
import config
from video_processor import split_video_into_frames
from face_detection import load_yolo_model, detect_faces_yolo, visualize_yolo_detections
from face_analysis import get_face_embedding, analyze_emotions, match_known_face, load_known_faces
from gaze import load_gaze_model, run_gaze_estimation
from profile_manager import update_profiles, recluster_profiles
from visualization import visualize_all
from llm_output import create_llm_input


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


# --- Helper Function for Processing a Single Frame ---
def _process_single_frame(img_path, frame_idx, yolo_model, known_faces, raw_profiles, frame_dimensions):
    """Loads image, detects faces, processes each face, returns frame data."""
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

        # 1. Raw YOLO Detection
        # Use verbose=False to reduce console spam from YOLO
        results = yolo_model(np_image, verbose=False)
        detected_boxes_data = [] # Store tuples of (box, conf, class_id)

        if results and results[0].boxes is not None:
            boxes_tensor = results[0].boxes.xyxy # xyxy format
            confs_tensor = results[0].boxes.conf
            classes_tensor = results[0].boxes.cls

            # Filter by initial confidence threshold and class (optional, but good practice)
            person_class_id = 0 # Assuming COCO class ID for person is 0
            for i in range(len(boxes_tensor)):
                conf = confs_tensor[i].item()
                class_id = int(classes_tensor[i].item())
                # Filter for 'person' class AND confidence threshold
                if class_id == person_class_id and conf >= config.YOLO_CONF_THRESHOLD:
                     box = boxes_tensor[i] # Keep as tensor for NMS
                     detected_boxes_data.append((box, conf)) # Store box tensor and conf

        print(f"  Frame {frame_idx+1}: Initial 'person' detections = {len(detected_boxes_data)}")

        # 2. Apply Non-Maximum Suppression (NMS) if detections exist
        final_boxes_pixels = [] # List to store boxes kept after NMS
        if detected_boxes_data:
            # Prepare data for torchvision.ops.nms
            nms_boxes = torch.stack([data[0] for data in detected_boxes_data]) # Stack box tensors
            nms_scores = torch.tensor([data[1] for data in detected_boxes_data]) # Create score tensor

            # Apply NMS
            keep_indices = torchvision.ops.nms(nms_boxes, nms_scores, config.NMS_IOU_THRESHOLD)

            # Get the boxes that were kept
            final_boxes_tensor = nms_boxes[keep_indices]
            final_boxes_pixels = [[int(coord) for coord in box.tolist()] for box in final_boxes_tensor] # Convert to list of lists of ints

            print(f"  Frame {frame_idx+1}: Detections after NMS = {len(final_boxes_pixels)}")


        # Optional: Visualize raw YOLO detections (can be slow)
        if config.SHOW_YOLO_DETECTIONS and detected_boxes_data:
             annotated_img = results[0].plot() # Plot original unfiltered results for comparison
             plt.figure(figsize=(8, 6))
             plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
             plt.title(f"Raw YOLO Detections (Before NMS): Frame {frame_idx+1}")
             plt.axis("off")
             plt.show(block=False)
             plt.pause(0.1)
             plt.close('all')

        # 3. Process Each *Final* Detected Face
        if final_boxes_pixels:
            # Prepare image tensor for gaze *once* per frame if faces survived NMS
            image_tensor = config.gaze_transform(pil_image).unsqueeze(0)

            for box_pixels in final_boxes_pixels: # Iterate over boxes kept by NMS
                x1, y1, x2, y2 = box_pixels # Already integers
                # Clip coordinates (should be redundant if YOLO output is valid, but safe)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x1 >= x2 or y1 >= y2: continue # Skip invalid box dimensions

                face_roi_cv = cv_image[y1:y2, x1:x2]
                if face_roi_cv.size == 0: continue

                # Process this single face using the helper
                face_data, raw_profiles = _process_single_face(
                    face_roi_cv, (x1, y1, x2, y2), frame_idx, known_faces, raw_profiles
                )

                if face_data: # If processing was successful
                    frame_faces_details.append(face_data)
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
            final_id = profile_assignment_map.get(temp_id)
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


# --- Main Analysis Function (Orchestrator) ---
def analyze_video_frames(image_paths, gaze_model, yolo_face_model, known_faces=None):
    """Orchestrates the frame-by-frame analysis, gaze, clustering, and finalization."""
    start_time = time.time()
    overall_results = {}
    raw_profiles = {} # Tracks faces across frames before reclustering
    all_frame_face_details = [] # List of lists: [[face1_data, face2_data], [face3_data], ...]
    all_norm_boxes = [] # List of lists: [[norm_box1, norm_box2], [norm_box3], ...]
    all_image_tensors = [] # List of tensors or None
    frame_dimensions = {} # Cache frame dimensions

    print(f"\nStarting analysis of {len(image_paths)} frames...")
    if not image_paths: return {}, {} # Handle empty input

    # --- Stage 1: Process Each Frame Individually ---
    for frame_idx, img_path in enumerate(image_paths):
        print(f"\rProcessing Frame {frame_idx+1}/{len(image_paths)}...", end="")
        # Call helper to process this frame
        frame_details, norm_boxes, img_tensor, raw_profiles, success = _process_single_frame(
            img_path, frame_idx, yolo_model, known_faces, raw_profiles, frame_dimensions
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
    # Call the final helper function
    final_frame_results, visualizations = _integrate_gaze_and_finalize(
        image_paths, all_frame_face_details, gaze_results, all_image_tensors,
        profile_assignment_map, final_profiles, frame_dimensions
    )

    # Combine results
    overall_results = final_frame_results # Add the frame data
    overall_results["profiles"] = final_profiles
    overall_results["visualizations"] = visualizations

    # --- Stage 5: Generate LLM Input ---
    print("\n--- Stage 5: Generating LLM Input JSON ---")
    llm_data = create_llm_input(overall_results, config.LLM_OUTPUT_PATH)

    end_time = time.time()
    print(f"\nAnalysis complete. Total time: {end_time - start_time:.2f} seconds.")

    return overall_results, llm_data


# (Keep display_analysis_results function as it was)
def display_analysis_results(results):
    """Displays the generated visualizations using Matplotlib."""
    visual_out = results.get("visualizations", {})
    if not visual_out:
        print("\nNo visualizations were generated or found in results.")
        # return # Keep going to print profile summary

    print("\nDisplaying Visualizations (Close each window to proceed)...")
    for path, vis_img in visual_out.items():
        try:
            plt.figure(figsize=(14, 10)) # Larger figure size
            plt.imshow(vis_img)
            plt.title(f"Analysis: {os.path.basename(path)}")
            plt.axis("off")
            plt.tight_layout()
            plt.show() # Will block until window is closed
        except Exception as e:
            print(f"Error displaying visualization for {os.path.basename(path)}: {e}")

    profs = results.get("profiles", {})
    if profs:
        print("\n--- Final Profile Summary ---")
        # Sort profiles by their final ID (key) before printing
        for pid, prof in sorted(profs.items()):
            frame_indices = prof.get('frames_seen', [])
            appearance_count = len(frame_indices)
            print(f"  Profile ID {pid} ({prof.get('name', 'Unknown')}): Appeared in {appearance_count} frame(s)")
            # Optional: Print frame indices if list is not too long
            # if appearance_count > 0 and appearance_count < 20:
            #    print(f"    Frame Indices: {frame_indices}")
    else:
        print("\nNo distinct profiles were finalized.")


# -----------------------------
# Main Execution Block
# -----------------------------
if __name__ == "__main__":

    print("--- Starting Video Analysis Pipeline ---")

    # --- 1. Extract Frames from Video ---
    print(f"\nStep 1: Extracting frames from video: {config.VIDEO_INPUT_PATH}")
    print(f"        Outputting to: {config.FRAME_OUTPUT_FOLDER}")
    if not os.path.exists(config.VIDEO_INPUT_PATH):
         print(f"FATAL ERROR: Input video file not found at '{config.VIDEO_INPUT_PATH}'")
         print("Please check the VIDEO_INPUT_PATH in config.py")
         exit(1)

    image_paths = split_video_into_frames(
        config.VIDEO_INPUT_PATH,
        config.FRAME_OUTPUT_FOLDER,
        frames_per_second=config.FRAMES_PER_SECOND_TO_EXTRACT
    )

    if not image_paths:
        print("\nError: Frame extraction failed or produced no frames.")
        exit(1)
    print(f"Step 1 Complete: Successfully extracted {len(image_paths)} frames.")

    # --- 2. (Optional) Load Known Faces ---
    print("\nStep 2: Loading models and optional known faces...")
    known_face_images = None # Set to None or a dict like {"Name": "path/to/img.jpg"}
    known_faces_embeddings = None

    # --- 3. Load Analysis Models ---
    yolo_model = None
    gaze_model = None
    try:
        yolo_model = load_yolo_model()
        gaze_model = load_gaze_model()
    except Exception as e:
        print(f"\nFATAL ERROR: Failed to load models. Exiting. Error: {e}")
        exit(1)

    if known_face_images and yolo_model:
        print("   Loading known face embeddings...")
        known_faces_embeddings = load_known_faces(known_face_images, yolo_model)
        if not known_faces_embeddings: print("   Warning: Known faces specified, but no embeddings loaded.")
    elif known_face_images: print("   Warning: Cannot load known faces - YOLO model failed.")

    print("Step 2 & 3 Complete: Models loaded.")

    # --- 4. Run Core Analysis (using the orchestrator function) ---
    print(f"\nStep 4: Analyzing {len(image_paths)} extracted frames...")
    analysis_results, llm_json_data = analyze_video_frames(
        image_paths,
        gaze_model,
        yolo_model,
        known_faces=known_faces_embeddings
    )
    print("Step 4 Complete: Frame analysis finished.")

    # --- 5. Display Results ---
    print("\nStep 5: Displaying analysis results...")
    display_analysis_results(analysis_results)
    print("Step 5 Complete: Results displayed.")

    print(f"\n--- Pipeline Finished ---")
    print(f"Visualizations saved to: {config.VISUALIZATION_OUTPUT_DIR}")
    print(f"LLM data saved to: {config.LLM_OUTPUT_PATH}")
    print("-----------------------------")
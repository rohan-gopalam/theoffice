import json
import numpy as np
import os  # Adding the missing import
from utils import NumpyEncoder # Use the custom encoder

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist() # Convert arrays to lists
        return super(NpEncoder, self).default(obj)


def create_llm_input(overall_results, output_path):
    """
    Transforms the analysis results into a JSON structure suitable for LLMs,
    ensuring necessary IDs and bounding boxes are included per person per frame.

    Args:
        overall_results (dict): The dictionary containing 'profiles' and
                                frame data keyed by image paths (e.g., result
                                from analyze_video_frames_with_tracking).
        output_path (str): The path where the JSON file should be saved.

    Returns:
        dict: The generated dictionary representing the LLM input JSON.
    """
    llm_data = {}
    frame_analysis_list = []

    # --- Create Session Summary ---
    # (Assuming 'profiles' key exists in overall_results)
    profiles_summary = {}
    final_profiles = overall_results.get("profiles", {})
    for profile_id, profile_data in final_profiles.items():
        profiles_summary[profile_id] = {
            "assigned_name": profile_data.get("name", f"Person {profile_id}"),
            "appeared_in_frame_indices": profile_data.get("frames_seen", [])
            # Add other summary info if needed
        }

    llm_data["session_summary"] = {
        "total_frames_processed": len(overall_results) - 1, # Subtract 1 for 'profiles' key
        "distinct_people_identified": profiles_summary
    }


    # --- Create Frame-by-Frame Analysis List ---
    # Iterate through the frames in overall_results
    # Sort keys to ensure consistent frame order if keys are filenames/paths
    frame_keys = sorted([k for k in overall_results.keys() if k != "profiles" and k != "visualizations"])

    for img_path in frame_keys:
        frame_data = overall_results.get(img_path, {})
        faces_in_frame_data = frame_data.get("faces", [])
        people_output_list = []

        # Process each face found in this frame
        for face_obj in faces_in_frame_data:
            # *** Extract the required data ***
            person_output = {
                "name": face_obj.get("name"),
                "profile_id": face_obj.get("profile_id"), # <<< FINAL CLUSTERED ID
                "tracker_id": face_obj.get("yolo_track_id"), # <<< ORIGINAL TRACKER ID
                "bbox_pixels": face_obj.get("bbox_pixels"), # <<< BOUNDING BOX
                "emotion_detected": face_obj.get("emotion"), # Assuming emotion is stored directly
                "gaze_info": { # Reconstruct gaze info if needed
                    "looking_at_camera": face_obj.get("gaze", {}).get("looking_at_camera"),
                    "inout_score": face_obj.get("gaze", {}).get("inout_score"),
                    # Add heatmap if you want it in the final JSON
                    # "heatmap": face_obj.get("gaze", {}).get("heatmap")
                },
                # Add other relevant fields like known_identity_match if desired
                "known_identity_match": face_obj.get("known_identity_match"),
                "known_identity_score": face_obj.get("known_identity_score"),
            }
            # Clean up None values if desired (optional)
            # person_output = {k: v for k, v in person_output.items() if v is not None}
            people_output_list.append(person_output)

        # Create the entry for this frame
        frame_entry = {
            # Use basename if img_path is a full path
            "frame_identifier": os.path.basename(img_path),
            "people_in_frame": people_output_list
        }
        frame_analysis_list.append(frame_entry)

    llm_data["frame_by_frame_analysis"] = frame_analysis_list

    # --- Add Chunk Metadata (if available/passed) ---
    # This part depends on how you handle chunking; it might be added
    # later in the process_video_in_chunks function as you currently do.
    # llm_data["chunk_metadata"] = overall_results.get("chunk_metadata", {})


    # --- Save the JSON ---
    try:
        # Use NpEncoder if you include NumPy arrays (like heatmaps)
        with open(output_path, 'w') as f:
            json.dump(llm_data, f, indent=2, cls=NpEncoder)
        print(f"Successfully saved LLM input JSON to {output_path}")
    except TypeError as e:
        print(f"Error saving JSON (possibly due to non-serializable data like NumPy arrays without NpEncoder): {e}")
        # Fallback: try saving without NpEncoder if you didn't include arrays
        try:
            with open(output_path, 'w') as f:
                 json.dump(llm_data, f, indent=2)
            print(f"Successfully saved LLM input JSON to {output_path} (without NpEncoder)")
        except Exception as e_inner:
            print(f"Error saving JSON even without NpEncoder: {e_inner}")

    except Exception as e:
        print(f"An error occurred saving the JSON: {e}")

    return llm_data # Return the dictionary as well

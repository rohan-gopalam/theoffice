# llm_output.py
import json
import numpy as np
import os
from utils import NumpyEncoder # Use the custom encoder

def create_llm_input(results_data, output_path="llm_analysis_input.json"):
    """
    Convert analysis results to a structured JSON file for LLM consumption,
    omitting internal profile IDs from the per-frame 'people' list.

    Args:
        results_data: The main analysis result dictionary containing 'profiles'
                      and frame-by-frame data.
        output_path: Path to save the JSON file.

    Returns:
        The structured data dictionary suitable for LLM input.
    """
    llm_data = {
        "session_summary": {
            "total_frames_processed": len([k for k in results_data.keys() if k not in ['profiles', 'visualizations']]),
            "distinct_people_identified": {} # Summary based on FINAL profiles
        },
        "frame_by_frame_analysis": []
    }

    # Populate session summary with final profile information
    final_profiles = results_data.get('profiles', {})
    for final_pid, prof_info in final_profiles.items():
        # Use the final assigned ID (e.g., 1, 2, 3...) as the key in the summary
        llm_data["session_summary"]["distinct_people_identified"][str(final_pid)] = {
            "assigned_name": prof_info.get('name', f"Person {final_pid}"),
            "appeared_in_frame_indices": prof_info.get('frames_seen', [])
            # Add more summary stats here if needed (e.g., dominant emotion over time)
        }

    # Populate frame-by-frame analysis
    for frame_path, frame_data in results_data.items():
        if frame_path in ['profiles', 'visualizations']:
            continue # Skip metadata keys

        frame_entry = {
            "frame_identifier": os.path.basename(frame_path), # Use filename as identifier
            "people_in_frame": []
        }

        for face_info in frame_data.get('faces', []):
            # Map the final profile ID back to the name for this person entry
            final_profile_id = face_info.get('profile_id') # This should be the FINAL ID after clustering
            person_name = "Unknown"
            if final_profile_id is not None and final_profile_id in final_profiles:
                 person_name = final_profiles[final_profile_id].get('name', f"Person {final_profile_id}")


            person_data = {
                # IMPORTANT: Do NOT include the raw 'profile_id' here for the LLM
                "name": person_name,
                "emotion_detected": face_info.get('emotion', "Unknown"),
                "bounding_box": { # Using pixel coordinates for clarity
                    "x1": int(face_info['bbox_pixels'][0]),
                    "y1": int(face_info['bbox_pixels'][1]),
                    "x2": int(face_info['bbox_pixels'][2]),
                    "y2": int(face_info['bbox_pixels'][3])
                }
                # Add known identity match score if relevant
                # "known_identity_match_score": face_info.get("similarity_score")
            }

            # Add gaze information if available
            gaze_info = face_info.get('gaze', {})
            if gaze_info:
                person_data["gaze_info"] = {
                    "is_looking_at_camera": gaze_info.get("looking_at_camera", False),
                    "camera_look_confidence": float(gaze_info.get("inout_score", 0.0))
                }
                # Optionally add normalized gaze target
                heatmap = gaze_info.get("heatmap")
                if heatmap is not None:
                    try:
                        heat_np = np.array(heatmap)
                        if heat_np.ndim == 2 and heat_np.size > 0:
                             max_idx = np.unravel_index(np.argmax(heat_np), heat_np.shape)
                             norm_x = float(max_idx[1]) / heat_np.shape[1]
                             norm_y = float(max_idx[0]) / heat_np.shape[0]
                             person_data["gaze_info"]["estimated_gaze_target_normalized"] = {"x": round(norm_x, 4), "y": round(norm_y, 4)}
                    except Exception as e:
                         print(f"Error processing heatmap for LLM output: {e}")

            frame_entry["people_in_frame"].append(person_data)

        # Add a simple natural language summary for the frame
        if frame_entry["people_in_frame"]:
            summaries = []
            for person in frame_entry["people_in_frame"]:
                summ = f"{person['name']} showing {person['emotion_detected']} emotion"
                if "gaze_info" in person:
                    look_status = "looking towards the camera" if person['gaze_info'].get("is_looking_at_camera") else "not looking towards the camera"
                    summ += f" ({look_status})"
                summaries.append(summ)
            frame_entry["natural_language_summary"] = ". ".join(summaries) + "."
        else:
            frame_entry["natural_language_summary"] = "No people detected in this frame."

        llm_data["frame_by_frame_analysis"].append(frame_entry)

    # Save the JSON file
    try:
        with open(output_path, 'w') as f:
            json.dump(llm_data, f, indent=2, cls=NumpyEncoder)
        print(f"LLM analysis input saved successfully to {output_path}")
        # Optional: Print a preview
        # preview = json.dumps(llm_data, indent=2, cls=NumpyEncoder)
        # print("\nLLM JSON Preview:")
        # print(preview[:1000] + "...\n" if len(preview) > 1000 else preview)
    except Exception as e:
        print(f"Error saving LLM JSON to {output_path}: {e}")

    return llm_data
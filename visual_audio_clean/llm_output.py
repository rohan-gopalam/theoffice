import json
import numpy as np
import os  # Adding the missing import
from utils import NumpyEncoder


def create_llm_input(results_data,  transcript=None, output_path="llm_analysis_input.json"):
    """
    Convert analysis results to a structured JSON file for LLM consumption,
    removing bounding boxes and natural language summaries. Correctly identifies
    gaze targets between people.

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
        "transcription of chunk": transcript,
        "frame_by_frame_analysis": []
    }

    # Populate session summary with final profile information
    final_profiles = results_data.get('profiles', {})
    for final_pid, prof_info in final_profiles.items():
        # Use the final assigned ID (e.g., 1, 2, 3...) as the key in the summary
        llm_data["session_summary"]["distinct_people_identified"][str(final_pid)] = {
            "assigned_name": prof_info.get('name', f"Person {final_pid}"),
            "appeared_in_frame_indices": prof_info.get('frames_seen', [])
        }


    # Populate frame-by-frame analysis
    for frame_path, frame_data in results_data.items():
        if frame_path in ['profiles', 'visualizations']:
            continue # Skip metadata keys

        frame_entry = {
            "frame_identifier": os.path.basename(frame_path), # Use filename as identifier
            "people_in_frame": []
        }



        # First pass: gather all people and their bounding boxes
        people_with_boxes = []
        for face_info in frame_data.get('faces', []):
            final_profile_id = face_info.get('profile_id')
            person_name = "Unknown"
            if final_profile_id is not None and final_profile_id in final_profiles:
                 person_name = final_profiles[final_profile_id].get('name', f"Person {final_profile_id}")

            # Extract emotion information - handle both string and dictionary formats
            emotion_data = face_info.get('emotion', "Unknown")
            
            # Store person with their bounding box for gaze target detection
            people_with_boxes.append({
                "name": person_name,
                "emotion_detected": emotion_data,
                "gaze_info": face_info.get('gaze', {}),
                "bbox": face_info['bbox_pixels'],  # Keep bbox temporarily for gaze calculation
                "profile_id": final_profile_id
            })

        # Second pass: create final output with gaze targets
        for person in people_with_boxes:
            # Prepare person data without bounding box
            person_data = {
                "name": person["name"],
                "emotion_detected": person["emotion_detected"],
                "gaze_info": None  # Default value
            }

            # Add gaze target information if available
            gaze_info = person.get("gaze_info", {})
            if gaze_info:
                gaze_data = {}
                
                # Only mark as looking at camera if confidence is very high
                camera_confidence_threshold = 0.5  # Strict threshold to avoid false positives
                if ("looking_at_camera" in gaze_info and 
                    gaze_info["looking_at_camera"] and 
                    gaze_info.get("inout_score", 0) > camera_confidence_threshold):
                    gaze_data["gaze_target"] = "camera"
                    
                # Check if looking at another person based on unnormalized gaze target
                elif "estimated_gaze_target_normalized" in gaze_info:
                    target = gaze_info["estimated_gaze_target_normalized"]
                    
                    # Get unnormalized coordinates
                    if isinstance(target, dict) and "x" in target and "y" in target:
                        # Calculate reasonable image dimensions
                        max_x = max(p["bbox"][2] for p in people_with_boxes) + 100  # Add padding
                        max_y = max(p["bbox"][3] for p in people_with_boxes) + 100  # Add padding
                        
                        # Convert normalized to absolute coordinates
                        gaze_x = int(target["x"] * max_x)
                        gaze_y = int(target["y"] * max_y)
                        
                        # Store the unnormalized coordinates
                        gaze_data["unnormalized_gaze_point"] = {"x": gaze_x, "y": gaze_y}
                        
                        # Look for a person target using expanded bounding boxes
                        person_target_found = False
                        for other_person in people_with_boxes:
                            if other_person["name"] == person["name"]:
                                continue  # Skip self
                            
                            # Get bounding box
                            x1, y1, x2, y2 = other_person["bbox"]
                            box_width = x2 - x1
                            box_height = y2 - y1
                            
                            # Expand by 25% in each direction for more forgiving detection
                            x1_expanded = max(0, x1 - int(box_width * 0.25))
                            y1_expanded = max(0, y1 - int(box_height * 0.25))
                            x2_expanded = min(max_x, x2 + int(box_width * 0.25))
                            y2_expanded = min(max_y, y2 + int(box_height * 0.25))
                            
                            # Check if gaze point is in expanded box
                            if x1_expanded <= gaze_x <= x2_expanded and y1_expanded <= gaze_y <= y2_expanded:
                                gaze_data["gaze_target"] = other_person["name"]
                                gaze_data["target_profile_id"] = other_person["profile_id"]
                                person_target_found = True
                                break
                        
                        # If no specific person target found, set to "other"
                        if not person_target_found:
                            gaze_data["gaze_target"] = "other"
                    else:
                        gaze_data["gaze_target"] = "other"
                elif "heatmap" in gaze_info:
                    # Handle heatmap-based gaze similarly to above
                    heatmap = gaze_info["heatmap"]
                    if isinstance(heatmap, (np.ndarray, list)):
                        heatmap_np = np.array(heatmap)
                        if heatmap_np.ndim == 2 and heatmap_np.size > 0:
                            max_idx = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
                            norm_y, norm_x = max_idx[0] / heatmap_np.shape[0], max_idx[1] / heatmap_np.shape[1]
                            
                            # Calculate reasonable image dimensions
                            max_x = max(p["bbox"][2] for p in people_with_boxes) + 100
                            max_y = max(p["bbox"][3] for p in people_with_boxes) + 100
                            
                            # Convert normalized to absolute coordinates
                            gaze_x = int(norm_x * max_x)
                            gaze_y = int(norm_y * max_y)
                            
                            # Store the unnormalized coordinates
                            gaze_data["unnormalized_gaze_point"] = {"x": gaze_x, "y": gaze_y}
                            
                            # Same person detection logic as above
                            person_target_found = False
                            for other_person in people_with_boxes:
                                if other_person["name"] == person["name"]:
                                    continue
                                
                                x1, y1, x2, y2 = other_person["bbox"]
                                box_width = x2 - x1
                                box_height = y2 - y1
                                
                                x1_expanded = max(0, x1 - int(box_width * 0.25))
                                y1_expanded = max(0, y1 - int(box_height * 0.25))
                                x2_expanded = min(max_x, x2 + int(box_width * 0.25))
                                y2_expanded = min(max_y, y2 + int(box_height * 0.25))
                                
                                if x1_expanded <= gaze_x <= x2_expanded and y1_expanded <= gaze_y <= y2_expanded:
                                    gaze_data["gaze_target"] = other_person["name"]
                                    gaze_data["target_profile_id"] = other_person["profile_id"]
                                    person_target_found = True
                                    break
                            
                            if not person_target_found:
                                gaze_data["gaze_target"] = "other"
                        else:
                            gaze_data["gaze_target"] = "other"
                    else:
                        gaze_data["gaze_target"] = "other"
                else:
                    gaze_data["gaze_target"] = "other"
                
                person_data["gaze_info"] = gaze_data

            frame_entry["people_in_frame"].append(person_data)

        llm_data["frame_by_frame_analysis"].append(frame_entry)

    # Save the JSON file
    try:
        with open(output_path, 'w') as f:
            json.dump(llm_data, f, indent=2, cls=NumpyEncoder)
        print(f"LLM analysis input saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving LLM JSON to {output_path}: {e}")

    return llm_data
import json
import os
import math # Needed for distance calculation
import re 
import argparse  # Needed for parsing frame numbers from keys

# --- Configuration ---
ANALYSIS_JSON_PATH = "visual_audio_clean/output/chunk_1/analysis_chunk_1.json" # Output from main.py
GROUND_TRUTH_PATH = "clip4file.json"
# --- Helper Function: Calculate Center ---
def calculate_center(box):
    """Calculates the center coordinates (x, y) of a bounding box."""
    if not box or len(box) != 4:
        print(f"Warning: Invalid bbox format for center calculation: {box}")
        return None
    try:
        # Ensure coordinates are numbers
        x1, y1, x2, y2 = map(float, box)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        return (center_x, center_y)
    except (ValueError, TypeError) as e:
        print(f"Warning: Error calculating center for bbox {box}: {e}")
        return None


# --- Helper Function: Calculate Distance ---
def calculate_distance(pointA, pointB):
    """Calculates the Euclidean distance between two points."""
    if not pointA or not pointB:
        return float('inf') # Return infinity if points are invalid
    return math.sqrt((pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2)

# --- Helper Function: Parse frame number from analysis identifier ---
def parse_frame_number_from_identifier(identifier_str):
    """
    Extracts the frame number from a frame_identifier string (e.g., 'frame_000630.jpg').
    Adjust the regex based on the actual identifier format.
    """
    if not identifier_str: return None
    base_key = os.path.basename(identifier_str) # Use basename just in case
    # Regex: Looks for digits after 'frame_' or '_' potentially followed by '.'
    match = re.search(r'(?:frame_|_)(\d+)\.?.*$', base_key)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    # Fallback: Try extracting last sequence of digits
    digits = re.findall(r'\d+', base_key)
    if digits:
        try:
            num_str = digits[-1]
            if len(num_str) < 9: # Basic check
                 return int(num_str)
        except ValueError:
            return None
    return None # Return None if no number could be parsed

# --- Core Logic Function ---
def find_closest_profile_by_distance(analysis_data, gt_entry):
    """
    Finds the profile_id in analysis_data for a specific frame that is
    closest to the ground truth target based on bounding box center distance.
    Uses closest frame matching based on index difference.

    Args:
        analysis_data (dict): The loaded analysis data (from main.py output).
        gt_entry (dict): A single entry from the ground truth data list.

    Returns:
        tuple: (closest_profile_id, min_distance, aligned_frame_identifier)
               Returns (None, float('inf'), None) if alignment fails or no faces are found.
    """
    closest_profile_id = None
    min_distance = float('inf')
    aligned_frame_identifier = None
    min_frame_diff = float('inf')
    analysis_frame_data_obj = None # Store the whole frame object

    gt_frame_index = gt_entry.get('label_frame_index')
    gt_target_tracker_id = gt_entry.get('zoom_target_id')

    if gt_frame_index is None or gt_target_tracker_id == -1:
        # print(f"Skipping GT entry: Missing frame index or target ID.") # Less verbose
        return None, float('inf'), None

    # 1. Frame Alignment - Find the closest frame in the analysis list
    analysis_frames_list = analysis_data.get("frame_by_frame_analysis", [])
    if not analysis_frames_list:
        # print("ERROR: 'frame_by_frame_analysis' list is missing or empty in analysis data.") # Less verbose
        return None, float('inf'), None

    found_valid_key = False
    for frame_obj in analysis_frames_list:
        identifier = frame_obj.get("frame_identifier")
        if not identifier:
            continue # Skip frames without identifier

        parsed_frame_num = parse_frame_number_from_identifier(identifier)
        if parsed_frame_num is not None:
            found_valid_key = True
            frame_diff = abs(parsed_frame_num - gt_frame_index)
            if frame_diff < min_frame_diff:
                min_frame_diff = frame_diff
                aligned_frame_identifier = identifier
                analysis_frame_data_obj = frame_obj # Store the closest frame object

    if not analysis_frame_data_obj or aligned_frame_identifier is None:
        # if not found_valid_key:
        #      print(f"ERROR: Could not parse frame numbers from ANY frame identifiers in analysis data.")
        # else:
        #      print(f"ERROR: Could not find a close enough frame identifier in analysis data for GT index {gt_frame_index}.")
        return None, float('inf'), None
    # else: # Less verbose success message
        # print(f"Frame Alignment SUCCESS: Closest analysis frame is '{aligned_frame_identifier}' (diff={min_frame_diff}) for GT index {gt_frame_index}.")

    # 2. Find Ground Truth BBox and Center
    gt_bbox = None
    for track in gt_entry.get('tracks_in_frame', []):
        # Compare IDs as strings for robustness
        if str(track.get('id')) == str(gt_target_tracker_id):
            gt_bbox = track.get('bbox')
            break

    if not gt_bbox:
        # print(f"ERROR: Could not find bbox for ground truth target ID {gt_target_tracker_id} in GT data for frame {gt_frame_index}.")
        return None, float('inf'), aligned_frame_identifier
    gt_center = calculate_center(gt_bbox)
    if not gt_center:
        # print(f"ERROR: Could not calculate center for ground truth bbox for frame {gt_frame_index}.")
        return None, float('inf'), aligned_frame_identifier
    # print(f"  Ground Truth Center: ({gt_center[0]:.2f}, {gt_center[1]:.2f}) for Tracker ID {gt_target_tracker_id}") # Optional

    # 3. Find Closest Face in Analysis Data based on Center Distance
    people_in_frame = analysis_frame_data_obj.get("people_in_frame", [])
    if not people_in_frame:
        # print(f"WARNING: No 'people_in_frame' found in analysis data for aligned frame '{aligned_frame_identifier}'.")
        return None, float('inf'), aligned_frame_identifier

    found_valid_analysis_face = False
    for person_obj in people_in_frame:
        analysis_profile_id = person_obj.get("profile_id") # Final clustered ID
        analysis_bbox = person_obj.get("bbox_pixels")     # BBox for this instance

        # Check if essential data is present
        if analysis_profile_id is None:
            # print(f"  Skipping person object due to missing 'profile_id': {person_obj.get('name', 'N/A')}")
            continue
        if analysis_bbox is None:
            # print(f"  Skipping person '{analysis_profile_id}' due to missing 'bbox_pixels'")
            continue

        analysis_center = calculate_center(analysis_bbox)
        if not analysis_center:
            # print(f"  Skipping person '{analysis_profile_id}' because center could not be calculated.")
            continue

        found_valid_analysis_face = True
        distance = calculate_distance(gt_center, analysis_center)
        # print(f"  - Profile ID: {analysis_profile_id}, Distance to GT: {distance:.2f}") # Optional

        if distance < min_distance:
            min_distance = distance
            closest_profile_id = analysis_profile_id # Store the final profile ID

    # if not found_valid_analysis_face: # Less verbose
         # print(f"WARNING: No valid faces (with profile_id and bbox_pixels) found in analysis frame '{aligned_frame_identifier}' to compare distance.")
         # return None, float('inf'), aligned_frame_identifier

    # Return None for profile_id if no valid face was found to compare against
    if not found_valid_analysis_face:
        return None, float('inf'), aligned_frame_identifier

    return closest_profile_id, min_distance, aligned_frame_identifier


# --- Main Alignment Function ---
def run_alignment(analysis_json_path, ground_truth_json_path):
    """
    Loads data, performs alignment for all ground truth entries, and returns results.

    Args:
        analysis_json_path (str): Path to the analysis JSON file.
        ground_truth_json_path (str): Path to the ground truth JSON file.

    Returns:
        list: A list of dictionaries, each containing the alignment result
              for one ground truth entry. Returns None if loading fails.
    """
    # --- Load Data ---
    if not os.path.exists(analysis_json_path):
        print(f"Error: Analysis JSON file not found at '{analysis_json_path}'")
        return None
    if not os.path.exists(ground_truth_json_path):
        print(f"Error: Ground truth JSON file not found at '{ground_truth_json_path}'")
        return None

    try:
        with open(analysis_json_path, 'r') as f:
            analysis_data = json.load(f)
        print(f"Loaded analysis data from {analysis_json_path}")
    except Exception as e:
        print(f"Error loading analysis data: {e}")
        return None

    try:
        with open(ground_truth_json_path, 'r') as f:
            ground_truth_data = json.load(f)
        print(f"Loaded ground truth data from {ground_truth_json_path}")
    except Exception as e:
        print(f"Error loading ground truth data: {e}")
        return None

    # --- Process all ground truth entries ---
    results = []
    print("\n--- Processing Ground Truth Entries ---")
    for i, gt_entry in enumerate(ground_truth_data):
        gt_frame_idx = gt_entry.get('label_frame_index')
        gt_target_id = gt_entry.get('zoom_target_id')

        if gt_frame_idx is None or gt_target_id == -1:
            continue # Skip entries without a valid target

        # --- Call the function to get the closest profile ID ---
        returned_profile_id, returned_distance, aligned_key = find_closest_profile_by_distance(
            analysis_data,
            gt_entry
        )

        result_entry = {
            "gt_frame_index": gt_frame_idx,
            "gt_target_tracker_id": gt_target_id,
            "aligned_analysis_frame": aligned_key,
            "closest_analysis_profile_id": returned_profile_id,
            "min_distance_pixels": round(returned_distance, 2) if returned_distance != float('inf') else None
        }
        results.append(result_entry)

        # Print result for this entry (optional, can be done by caller)
        # print(f"\nProcessed GT Frame {gt_frame_idx} (Target ID: {gt_target_id}):")
        # if returned_profile_id is not None:
        #     print(f"  -> Closest Profile ID: '{returned_profile_id}' (Distance: {result_entry['min_distance_pixels']} px)")
        #     print(f"     (Aligned Analysis Frame: '{aligned_key}')")
        # else:
        #     print(f"  -> Could not find a match. (Aligned Analysis Frame: '{aligned_key}')")

    print(f"--- Finished processing {len(results)} ground truth entries with targets ---")
    return results # Return the list of results

# --- Main Execution Block (if script is run directly) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align ground truth with analysis data and find the closest profile ID based on bounding box center distance.")
    parser.add_argument("-a", "--analysis", default=ANALYSIS_JSON_PATH, help=f"Path to the analysis JSON file. Default: {DEFAULT_ANALYSIS_JSON_PATH}")
    parser.add_argument("-g", "--groundtruth", default=GROUND_TRUTH_PATH, help=f"Path to the ground truth JSON file. Default: {DEFAULT_GROUND_TRUTH_PATH}")
    parser.add_argument("-o", "--output", help="Optional path to save the alignment results to a JSON file.")

    args = parser.parse_args()

    # Call the main alignment function
    alignment_results = run_alignment(args.analysis, args.groundtruth)

    # Print the results (or handle them as needed)
    if alignment_results is not None:
        print("\n--- Alignment Results ---")
        # Pretty print the first few results as an example
        import pprint
        pprint.pprint(alignment_results[:5])
        if len(alignment_results) > 5:
             print("...")

        # --- Optional: Save results to a file ---
        if args.output:
            output_filename = args.output
            try:
                with open(output_filename, 'w') as f_out:
                    json.dump(alignment_results, f_out, indent=2)
                print(f"\n--- Full alignment results saved to {output_filename} ---")
            except Exception as e:
                print(f"\nError saving results to {output_filename}: {e}")
        else:
             print("\n--- Use -o <filename.json> to save full results to a file. ---")

    else:
        print("\nAlignment process failed (check errors above).")


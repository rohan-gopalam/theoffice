import json
import os
import math # Needed for distance calculation
import re 
import argparse  # Needed for parsing frame numbers from keys
import closest_frame

# --- Configuration ---
DEFAULT_BASE_DIR = "/Users/anaishadas/Desktop/theoffice/visual_audio_clean/output" # Directory containing chunk_1, chunk_2, ...
DEFAULT_GROUND_TRUTH_PATH = "/Users/anaishadas/Desktop/theoffice/visual_audio_clean/clip4file.json"
DEFAULT_FPS = 30
DEFAULT_MAX_CHUNKS = 76
import json
import os
import math
import re
import argparse
import sys
import pprint # For example printing

# --- Helper Function: Calculate Center (Keep for potential future use) ---
def calculate_center(box):
    """Calculates the center coordinates (x, y) of a bounding box."""
    if not box or len(box) != 4:
        # print(f"Warning: Invalid bbox format for center calculation: {box}")
        return None
    try:
        x1, y1, x2, y2 = map(float, box)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        return (center_x, center_y)
    except (ValueError, TypeError) as e:
        print(f"Warning: Error calculating center for bbox {box}: {e}")
        return None

# --- Helper Function: Calculate Distance (Keep for potential future use) ---
def calculate_distance(pointA, pointB):
    """Calculates the Euclidean distance between two points."""
    if not pointA or not pointB:
        return float('inf')
    return math.sqrt((pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2)

# --- Helper Function: Parse frame number from analysis identifier ---
def parse_frame_number_from_identifier(identifier_str):
    """
    Extracts the frame number from a frame_identifier string (e.g., 'frame_000630.jpg').
    Adjust the regex based on the actual identifier format. Returns None on failure.
    """
    if not identifier_str: return None
    base_key = os.path.basename(identifier_str)
    match = re.search(r'(?:frame_|_)(\d+)\.?.*$', base_key)
    if match:
        try:
            return int(match.group(1))
        except ValueError: return None
    digits = re.findall(r'\d+', base_key)
    if digits:
        try:
            num_str = digits[-1]
            if len(num_str) < 9: return int(num_str)
        except ValueError: return None
    # print(f"Warning: Could not parse frame number from identifier: '{identifier_str}'") # Optional
    return None

# --- Helper Function: Calculate timestamp from frame number ---
def calculate_timestamp(frame_number, fps):
    """Calculates the timestamp in seconds given a frame number and FPS."""
    if frame_number is None or fps <= 0:
        return None
    return float(frame_number) / fps


def find_closest_id(aligned_frame_identifier, people_in_frame, gt_entry):
    gt_bbox = None
    gt_target_tracker_id = gt_entry.get('zoom_target_id')
    for track in gt_entry.get('tracks_in_frame', []):
        # Compare IDs as strings for robustness
        if str(track.get('id')) == str(gt_target_tracker_id):
            gt_bbox = track.get('bbox')
            break

    if not gt_bbox:
        # print(f"ERROR: Could not find bbox for ground truth target ID {gt_target_tracker_id} in GT data for frame {gt_frame_index}.")
        return gt_bbox, float('inf'), aligned_frame_identifier
    gt_center = calculate_center(gt_bbox)
    
    if not gt_center:
        # print(f"ERROR: Could not calculate center for ground truth bbox for frame {gt_frame_index}.")
        return None, float('inf'), aligned_frame_identifier
    # print(f"  Ground Truth Center: ({gt_center[0]:.2f}, {gt_center[1]:.2f}) for Tracker ID {gt_target_tracker_id}") # Optional

    frame_centers = []
    min_dist = 10000000
    min_dist_id = None
    for people in people_in_frame:
        f_bbox = people["bbox"]
        f_center = calculate_center(f_bbox)
        distance  = calculate_distance(f_center, gt_center)
        if (distance < min_dist):
            min_dist = distance
            min_dist_id = people["name"].split()[-1]
    return min_dist_id



# --- Function to find closest frame across chunks for a SINGLE timestamp ---
def find_closest_frame_for_timestamp(base_dir, target_time, fps, max_chunks, chunk_cache):
    """
    Searches through analysis JSON files in chunk directories to find the frame
    closest to the target timestamp. Uses a cache for loaded chunk data.

    Args:
        base_dir (str): The base directory containing chunk_N folders.
        target_time (float): The target timestamp in seconds.
        fps (float): The assumed FPS of the original video.
        max_chunks (int): The maximum chunk number to check.
        chunk_cache (dict): A dictionary to cache loaded analysis data.

    Returns:
        tuple: (best_chunk_number, best_frame_identifier, min_time_diff)
               Returns (None, None, float('inf')) if no suitable frame is found.
    """
    overall_min_time_diff = float('inf')
    best_chunk_number = None
    best_frame_identifier = None
    best_count = None
    found_any_chunk = False # Tracks if at least one chunk was processed

    # Iterate through potential chunk directories
    for chunk_num in range(1, max_chunks + 1):
        analysis_data = None
        # Check cache first
        if chunk_num in chunk_cache:
            analysis_data = chunk_cache[chunk_num]
            if analysis_data is None: # Explicitly marked as non-existent or failed
                 continue
            found_any_chunk = True
        else:
            # Load if not in cache
            chunk_dir = os.path.join(base_dir, f"chunk_{chunk_num}")
            analysis_file = os.path.join(chunk_dir, f"analysis_chunk_{chunk_num}.json")

            if not os.path.exists(analysis_file):
                 chunk_cache[chunk_num] = None # Mark as non-existent
                 if chunk_num == 1:
                      print(f"Error: Analysis file not found for chunk 1: {analysis_file}")
                 # Stop searching sequentially if a chunk is missing
                 break

            found_any_chunk = True
            try:
                with open(analysis_file, 'r') as f:
                    analysis_data = json.load(f)
                chunk_cache[chunk_num] = analysis_data # Store loaded data in cache
            except Exception as e:
                print(f"Warning: Error loading {analysis_file}: {e}. Skipping chunk.")
                chunk_cache[chunk_num] = None # Mark as failed
                continue

        # Process the loaded analysis data for this chunk
        analysis_frames_list = analysis_data.get("frame_by_frame_analysis", [])
        if not analysis_frames_list:
            continue # Skip if no frame data
        count = 0

        # Iterate through frames within this chunk's analysis file
        for frame_obj in analysis_frames_list:
            identifier = frame_obj.get("frame_identifier")
            if not identifier: continue

            parsed_frame_num = parse_frame_number_from_identifier(identifier)
            frame_time = calculate_timestamp(parsed_frame_num, fps)

            if frame_time is not None:
                time_diff = abs(frame_time - target_time)
                if time_diff < overall_min_time_diff:
                    overall_min_time_diff = time_diff
                    best_chunk_number = chunk_num
                    best_frame_identifier = identifier
                    best_count = count
            count += 1

    # Return None if no chunks were even found/processed
    if not found_any_chunk and best_chunk_number is None:
         # Print error only once if needed, maybe outside the loop
         # print(f"Error: No chunk directories/analysis files found starting from chunk_1 in '{base_dir}'.")
         return None, None, float('inf')

    return best_chunk_number, best_frame_identifier, overall_min_time_diff, best_count


# --- Main Callable Function ---
def run_alignment(analysis_base_dir, ground_truth_json_path, fps, max_chunks=100):
    """
    Loads data, performs alignment for all ground truth entries using timestamps,
    and returns results.

    Args:
        analysis_base_dir (str): Base directory containing chunk_N folders.
        ground_truth_json_path (str): Path to the ground truth JSON file.
        fps (float): Assumed FPS of the original video.
        max_chunks (int): Maximum chunk number to search through.

    Returns:
        list: A list of dictionaries, each containing the alignment result
              for one ground truth entry. Returns None if loading fails.
    """
    # --- Validate Inputs ---
    if not os.path.isdir(analysis_base_dir):
         print(f"Error: Analysis base directory not found or not a directory: '{analysis_base_dir}'")
         return None
    if not os.path.exists(ground_truth_json_path):
        print(f"Error: Ground truth JSON file not found at '{ground_truth_json_path}'")
        return None
    if fps <= 0:
        print("Error: FPS must be a positive number.")
        return None

    # --- Load Ground Truth Data ---
    try:
        with open(ground_truth_json_path, 'r') as f:
            ground_truth_data = json.load(f)
        print(f"Loaded ground truth data from {ground_truth_json_path}")
    except Exception as e:
        print(f"Error loading ground truth data: {e}")
        return None

    # --- Process all ground truth entries ---
    results = []
    chunk_data_cache = {} # Cache loaded analysis JSONs
    print(f"\n--- Processing Ground Truth Entries (using assumed FPS: {fps}) ---")
    gt_entries_with_time = 0
    found_matches = 0

    for i, gt_entry in enumerate(ground_truth_data):
        gt_time = gt_entry.get('label_time')
        gt_frame_idx = gt_entry.get('label_frame_index') # Keep for reference

        if gt_time is None:
            # Optionally report skipped entries
            # print(f"Skipping GT entry #{i+1}: Missing 'label_time'.")
            continue

        gt_entries_with_time += 1

        # --- Call the function to get the closest frame ---
        chunk, frame_id, time_diff, frame_ind = find_closest_frame_for_timestamp(
            analysis_base_dir,
            gt_time,
            fps,
            max_chunks,
            chunk_data_cache # Pass cache
        )

        if chunk is not None and frame_id is not None:
            found_matches += 1

        # Store the result for this ground truth timestamp
        result_entry = {
            "gt_frame_index": gt_frame_idx,
            "gt_time": gt_time,
            # Include GT target ID if needed for context, though not used for alignment
            "gt_target_tracker_id": gt_entry.get('zoom_target_id'),
            "closest_chunk": chunk,
            "closest_frame_identifier": frame_id,
            "time_difference_s": round(time_diff, 3) if time_diff != float('inf') else None        
            }
        results.append(result_entry)

        analysis_curr_dir = "/Users/anaishadas/Desktop/theoffice/visual_audio_clean/analysis_chunk_1.json"

        try:
            with open(analysis_curr_dir, 'r') as f:
                analysis_chunk_data = json.load(f)
                print(f"Loaded analysis data from {analysis_curr_dir}")
        except Exception as e:
                print(f"Error loading ground truth data: {e}")
                return None
    
        analysis_frames_list = analysis_chunk_data.get("frame_by_frame_analysis", [])
        analysis_curr_frame = analysis_frames_list[frame_ind]
        # print(analysis_curr_frame)

        people_in_frame = analysis_curr_frame.get("people_in_frame", [])
        print(people_in_frame)

        closest_id = find_closest_id(frame_id, people_in_frame, gt_entry)
        result_entry["closest_profile_id"] = closest_id
    




        

        if not os.path.exists(analysis_curr_dir):
            print(f"Error: Analysis Path file not found at '{analysis_curr_dir}'")
            return None

        # Optional: Print progress per entry
        # if (i + 1) % 10 == 0 or i == len(ground_truth_data) - 1:
        #      print(f"Processed GT entry {i+1}/{len(ground_truth_data)} (Time: {gt_time:.3f}s) -> Found: Chunk {chunk}, Frame '{frame_id}' (Diff: {result_entry['time_difference_s']}s)")

    print(f"--- Finished processing {gt_entries_with_time} ground truth entries with timestamps ---")
    print(f"--- Found closest frames for {found_matches} entries ---")
    return results # Return the list of results

# --- Main Execution Block (Example Usage if script is run directly) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="For each timestamp in a ground truth file, find the closest analysis frame across multiple chunk files.")
    parser.add_argument("-g", "--groundtruth", default=DEFAULT_GROUND_TRUTH_PATH, help=f"Path to the ground truth JSON file (e.g., clip4file.json). Default: '{DEFAULT_GROUND_TRUTH_PATH}'")
    parser.add_argument("-d", "--basedir", default=DEFAULT_BASE_DIR, help=f"Base directory containing chunk_N folders. Default: '{DEFAULT_BASE_DIR}'")
    parser.add_argument("-f", "--fps", type=float, default=DEFAULT_FPS, help=f"Assumed FPS of the original video for timestamp calculation. Default: {DEFAULT_FPS}")
    parser.add_argument("-m", "--maxchunks", type=int, default=DEFAULT_MAX_CHUNKS, help=f"Maximum chunk number to check. Default: {DEFAULT_MAX_CHUNKS}")
    parser.add_argument("-o", "--output", help="Optional path to save the timestamp-to-frame mapping results to a JSON file.")

    args = parser.parse_args()

    # Call the main alignment function
    # This is the function you would call from another script
    alignment_results = run_alignment(
        analysis_base_dir=args.basedir,
        ground_truth_json_path=args.groundtruth,
        fps=args.fps,
        max_chunks=args.maxchunks
    )

    # Example of how to handle the returned results
    if alignment_results is not None:
        print("\n--- Alignment Results Summary (First 5) ---")
        pprint.pprint(alignment_results[:5])
        if len(alignment_results) > 5:
             print("...")

        # --- Optional: Save results to a file ---
        if args.output:
            output_filename = args.output
            try:
                with open(output_filename, 'w') as f_out:
                    json.dump(alignment_results, f_out, indent=2)
                print(f"\n--- Full results saved to {output_filename} ---")
            except Exception as e:
                print(f"\nError saving results to {output_filename}: {e}")
        elif alignment_results: # Only suggest saving if there are results
             print("\n--- Use -o <filename.json> to save full results to a file. ---")

    else:
        print("\nAlignment process failed or produced no results (check errors above).")


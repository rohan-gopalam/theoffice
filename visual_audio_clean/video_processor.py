# video_processor.py
import cv2
import os
import math # Import math for ceiling function

def split_video_into_frames(video_path, output_folder, frames_per_second=1):
    """
    Splits a video into individual frames, saving them as images.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the folder where frames will be saved.
        frames_per_second (int/float): How many frames to extract per second of video.
                                        Set to 0 or None to extract all frames.

    Returns:
        list: A list of paths to the saved frame images, or an empty list if failed.
    """
    frame_paths = [] # List to store the paths of saved frames

    # Create the output folder if it doesn't exist
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")
    except OSError as e:
        print(f"Error creating directory {output_folder}: {e}")
        return frame_paths # Return empty list on directory creation error

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened correctly
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return frame_paths # Return empty list

    # Get video properties
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / native_fps if native_fps > 0 else 0

    print(f"Video Info: Path='{video_path}', FPS={native_fps:.2f}, Total Frames={total_frames}, Duration={duration:.2f}s")

    if native_fps <= 0:
        print("Warning: Could not determine video FPS. Extracting every frame.")
        frames_per_second = 0 # Fallback to extracting all frames

    # Determine frame interval
    if frames_per_second is not None and frames_per_second > 0:
        # Calculate the interval between frames to capture based on desired FPS
        frame_interval = native_fps / frames_per_second
        if frame_interval < 1:
            print(f"Warning: Desired FPS ({frames_per_second}) is higher than video FPS ({native_fps:.2f}). Extracting every frame.")
            frame_interval = 1 # Extract every frame if desired rate is too high
        else:
            frame_interval = int(round(frame_interval)) # Round to nearest integer frame skip
        print(f"Extracting approximately {frames_per_second} frame(s) per second (every ~{frame_interval} frame(s)).")
    else:
        frame_interval = 1 # Extract every frame
        print("Extracting every frame.")


    saved_frame_count = 0
    current_frame_index = 0

    while True:
        # Set the read position to the desired frame index
        # Using set(CAP_PROP_POS_FRAMES) can be inaccurate on some videos/codecs.
        # A more robust way for frame skipping is reading frames sequentially and only saving at intervals.
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)

        ret, frame = cap.read()

        if ret:
            # Generate filename using the saved frame count for sequential numbering
            frame_filename = os.path.join(output_folder, f"frame_pbm_{saved_frame_count:04d}.png")

            try:
                # Save the frame as an image
                cv2.imwrite(frame_filename, frame)
                frame_paths.append(frame_filename) # Add path to the list
                # Optional: Print progress less frequently
                if saved_frame_count % 10 == 0:
                     print(f"Saved frame {saved_frame_count} (Source index: {current_frame_index}) -> {frame_filename}")

                saved_frame_count += 1

            except Exception as e:
                 print(f"Error saving frame {current_frame_index} to {frame_filename}: {e}")

            # Advance to the next frame index to read
            current_frame_index += frame_interval

            # Stop if we've requested more frames than available
            if current_frame_index >= total_frames:
                 break

        else:
            # End of video stream or error reading frame
            print(f"Finished reading video or encountered read error after saving {saved_frame_count} frames.")
            break

    # Release the video capture object
    cap.release()
    print(f"Successfully extracted and saved {saved_frame_count} frames to {output_folder}.")

    return frame_paths # Return the list of paths


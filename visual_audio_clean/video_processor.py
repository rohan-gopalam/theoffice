# video_processor.py
import cv2
import os
import math # Import math for ceiling function


# In video_processor.py
def split_video_into_frames(video_path, output_folder, frames_per_second=None, start_time=None, end_time=None):
    """
    Extract frames from a video within a specific time range.
    
    Args:
        video_path: Path to the video file
        output_folder: Folder to save extracted frames
        frames_per_second: Frames to extract per second (use None for every frame)
        start_time: Start time in seconds (None for beginning)
        end_time: End time in seconds (None for end of video)
        
    Returns:
        List of paths to extracted frames
    """
    os.makedirs(output_folder, exist_ok=True)
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error opening video file {video_path}")
        return []
    
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    # Calculate frame indices for the time range
    start_frame = 0 if start_time is None else int(start_time * fps)
    end_frame = frame_count if end_time is None else int(end_time * fps)
    
    # Calculate frame interval based on desired FPS
    if frames_per_second and frames_per_second > 0 and frames_per_second < fps:
        frame_interval = int(fps / frames_per_second)
    else:
        frame_interval = 1  # Extract every frame
    
    # Seek to start frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_paths = []
    current_frame = start_frame
    
    while current_frame < end_frame:
        ret, frame = video.read()
        if not ret:
            break
        
        # Save this frame if it matches our interval
        if (current_frame - start_frame) % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{current_frame:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_paths.append(frame_filename)
        
        current_frame += 1
    
    video.release()
    print(f"Extracted {len(frame_paths)} frames from time {start_time or 0:.2f}s to {end_time or duration:.2f}s")
    return frame_paths
import subprocess
import os

# List of (video_path, label_json_path) pairs
video_label_pairs = [
    ("../caseoh.mp4", "../caseoh_labels.json"),
    ("../chugs.mp4", "../chugs.json"),
    ("../clip4.mp4", "../clip4file.json"),
    ("../kaiandkevin.mp4", "../kaiandkevin.json"),
    ("../kaicenat.mp4", "../kaicenat_labels.json"),
    ("../ray1.mp4", "../ray1.json"),
    ("../valky_charity.mp4", "../valky_charity.json"),
    ("../valky_drunk.mp4", "../valky_drunk.json"),

]
output_dir = "batch_outputs"
os.makedirs(output_dir, exist_ok=True)

for idx, (video_path, label_path) in enumerate(video_label_pairs, 1):
    base_video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n=== Processing Pair {idx}: {base_video_name} ===")

    # Run main.py
    print(f"-> Running main.py on {video_path}")
    subprocess.run(["python", "main.py", video_path], check=True)

    # Path where main.py saves output chunks
    chunks_dir = os.path.join("output", base_video_name)

    # Run process_output.py with 2 arguments: labeler JSON, chunk output path
    print(f"-> Running process_output.py on {label_path} and {chunks_dir}")
    subprocess.run(["python", "process_output.py", label_path, chunks_dir], check=True)

    # Rename output_filtered_<name>.json and move to batch_outputs/
    filtered_json_name = f"output_filtered_{os.path.splitext(os.path.basename(label_path))[0]}.json"
    filtered_json_path = os.path.join(os.path.dirname(label_path), filtered_json_name)

    if os.path.exists(filtered_json_path):
        new_path = os.path.join(output_dir, filtered_json_name)
        os.rename(filtered_json_path, new_path)
        print(f"Saved filtered output to {new_path}")
    else:
        print(f"⚠️ Warning: Expected {filtered_json_name} not found.")











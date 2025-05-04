import os
import sys
import json
import math
from collections import defaultdict, Counter
from visual_audio_clean.alignment import run_alignment


output_path = "/Users/aditya/Documents/code projects/theoffice/output/clip4.mp4/"

def get_times_by_zoom_id(data):
    times_by_id = defaultdict(list)
    for e in data:
        zid = e.get('zoom_target_id')
        times_by_id[zid].append(e['label_time'])
    return times_by_id

def detect_fps_from_chunks(base_dir, maxchunks=100):
    """
    Scan chunk_1, chunk_2, … until you find an analysis JSON,
    then return its 'fps' field.
    """
    for i in range(1, maxchunks+1):
        path = os.path.join(base_dir, f"chunk_{i}", f"analysis_chunk_{i}.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                meta = json.load(f)
            fps = meta.get('fps')
            if fps is not None:
                return float(fps)
    raise RuntimeError(f"No FPS found in any of the first {maxchunks} chunks")

def fill_per_second(chunks, data_path, base_fps=30, maxchunks=100):
    """
    Treat each JSON entry as a switch at its label_time.
    Return a dict mapping each integer second → current zoom_target_id.
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    fps = detect_fps_from_chunks(chunks)
    
    alignment_results = run_alignment(
        analysis_base_dir=chunks,
        ground_truth_json_path=data_path,
        fps = fps,
        max_chunks = maxchunks
    )
    
    with open(data_path, 'r') as f:
        data = json.load(f)
        
    markers = sorted(
        [(e['label_time'], e['zoom_target_id']) for e in data],
        key=lambda x: x[0]
    )

    # ensure coverage from t=0
    if markers[0][0] > 0:
        markers.insert(0, (0.0, markers[0][1]))

    last_sec = math.ceil(markers[-1][0])
    timeline = {}
    idx = 0

    for sec in range(0, last_sec + 1):
        while idx + 1 < len(markers) and markers[idx + 1][0] <= sec:
            idx += 1
        timeline[sec] = markers[idx][1]

    return timeline

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python process_output.py path/to/labeler.json path/to/output_chunks_dir")
        sys.exit(1)

    json_path = sys.argv[1]
    chunks_dir = sys.argv[2]

    # Load and display stats
    with open(json_path, 'r') as f:
        data = json.load(f)

    times_by_id = get_times_by_zoom_id(data)
    print("All times by zoom_target_id:")
    for zid, tlist in sorted(times_by_id.items()):
        print(f"  id={zid:<5} → {len(tlist)} events")

    # Build per-second timeline
    sec_timeline = fill_per_second(chunks_dir, json_path)

    # Output path: output_filtered_<basename>.json
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    out_path = os.path.join(os.path.dirname(json_path), f"output_filtered_{base_name}.json")

    with open(out_path, 'w') as out_f:
        json.dump(sec_timeline, out_f, indent=2)

    print(f"\nPer-second timeline saved to {out_path}")

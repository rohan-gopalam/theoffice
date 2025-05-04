import os
import sys
import json
import math
from collections import defaultdict, Counter

def get_times_by_zoom_id(data):
    times_by_id = defaultdict(list)
    for e in data:
        zid = e.get('zoom_target_id')
        times_by_id[zid].append(e['label_time'])
    return times_by_id

def fill_per_second(data):
    """
    Treat each JSON entry as a switch at its label_time.
    Return a dict mapping each integer second → current zoom_target_id.
    """
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
    here = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) == 2:
        json_path = sys.argv[1]
    else:
        json_path = os.path.join(here, 'caseoh_labels.json')

    # load markers
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1) (optional) print full time lists
    times_by_id = get_times_by_zoom_id(data)
    print("All times by zoom_target_id:")
    for zid, tlist in sorted(times_by_id.items()):
        print(f"  id={zid:<5} → {len(tlist)} events")

    # 2) build per-second timeline
    sec_timeline = fill_per_second(data)

    # 3) save to output_filtered.json
    out_path = os.path.join(here, 'output_filtered.json')
    with open(out_path, 'w') as out_f:
        json.dump(sec_timeline, out_f, indent=2)

    print(f"\nPer-second timeline saved to {out_path}")

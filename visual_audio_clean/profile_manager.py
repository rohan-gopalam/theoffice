# profile_manager.py
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cosine
import config

def update_profiles(detected_face_embedding, current_profiles, frame_idx, embedding_threshold=config.EMBEDDING_THRESHOLD):
    """
    Matches a detected face embedding to existing profiles or creates a new one.

    Args:
        detected_face_embedding: NumPy array of the face embedding.
        current_profiles: Dict {profile_id: {"embedding": emb, "frames_seen": [idx]}}.
        frame_idx: The index of the current frame.
        embedding_threshold: Cosine distance threshold for matching.

    Returns:
        Tuple: (profile_id, updated_profiles)
    """
    matched_pid = None
    min_dist = float('inf')

    # Find the best match among existing profiles
    for pid, profile_data in current_profiles.items():
        # Check if embeddings are valid before calculating distance
        if profile_data.get("embedding") is not None and detected_face_embedding is not None:
            try:
                dist = cosine(profile_data["embedding"], detected_face_embedding)
                if dist < embedding_threshold and dist < min_dist:
                    min_dist = dist
                    matched_pid = pid
            except Exception as e:
                 print(f"Error comparing embedding for profile {pid}: {e}")
                 # Decide how to handle: continue, assign new ID, etc.
                 # For now, let's skip this profile for matching
                 continue
        else:
            # Handle cases where an embedding might be None (e.g., previous error)
            # print(f"Warning: Skipping profile {pid} due to missing embedding.")
            pass


    if matched_pid is not None:
        # Found a match, update the profile's frame list
        current_profiles[matched_pid]["frames_seen"].append(frame_idx)
        # Optionally, update the embedding? (e.g., average) - Simple approach: keep first
        return matched_pid, current_profiles
    else:
        # No match found, create a new profile
        new_pid = len(current_profiles) + 1
        current_profiles[new_pid] = {
            "embedding": detected_face_embedding,
            "frames_seen": [frame_idx],
            "name": f"Person {new_pid}" # Initial generic name
        }
        return new_pid, current_profiles


def recluster_profiles(profiles, eps=config.DBSCAN_EPS):
    """
    Recluster profile embeddings using DBSCAN based on cosine distance.

    Args:
        profiles: Dictionary {profile_id: {"embedding": emb, "frames_seen": [...]}}.
        eps: DBSCAN epsilon distance threshold.

    Returns:
        Tuple: (new_profile_assignments, final_profiles_summary)
            - new_profile_assignments: Dict mapping original profile_id to new cluster label (int).
                                       Noise points (rare with min_samples=1) might get -1.
            - final_profiles_summary: Dict {cluster_label: {"name": str, "frames_seen": list}}.
                                       Embeddings are not stored here, only summary info.
    """
    print(f"\nReclustering profiles using DBSCAN with eps={eps}...")
    if not profiles:
        print("No profiles to recluster.")
        return {}, {}

    profile_ids = list(profiles.keys())
    embeddings = [profiles[pid]["embedding"] for pid in profile_ids if profiles[pid].get("embedding") is not None]
    valid_profile_ids = [pid for pid in profile_ids if profiles[pid].get("embedding") is not None]

    if not embeddings:
        print("No valid embeddings found for reclustering.")
        return {}, {}

    embeddings = np.array(embeddings)
    if embeddings.ndim != 2:
        print(f"Error: Embeddings array has unexpected shape {embeddings.shape}. Cannot cluster.")
        # Create a dummy mapping where each valid profile is its own cluster
        new_profile_assignments = {pid: i for i, pid in enumerate(valid_profile_ids)}
        final_profiles_summary = {
            i: {"name": profiles[pid].get("name", f"Person {i+1}"), "frames_seen": profiles[pid]["frames_seen"]}
            for i, pid in enumerate(valid_profile_ids)
        }
        return new_profile_assignments, final_profiles_summary


    # Use DBSCAN with cosine metric
    db = DBSCAN(eps=eps, min_samples=1, metric='cosine').fit(embeddings)
    labels = db.labels_ # Cluster labels for each embedding (-1 for noise)

    new_profile_assignments = {old_pid: label for old_pid, label in zip(valid_profile_ids, labels)}

    # Aggregate results for final summary
    final_profiles_summary = {}
    for old_pid, cluster_label in new_profile_assignments.items():
        if cluster_label == -1: # Handle noise points if they occur (assign unique ID)
             unique_id = max(final_profiles_summary.keys()) + 1 if final_profiles_summary else 0
             final_profiles_summary[unique_id] = {
                 "name": profiles[old_pid].get("name", f"Person {unique_id+1}"), # Use original name if available
                 "frames_seen": sorted(list(set(profiles[old_pid]["frames_seen"])))
             }
             new_profile_assignments[old_pid] = unique_id # Update assignment map
             print(f"Profile {old_pid} treated as noise, assigned new ID {unique_id+1}")
             continue

        if cluster_label not in final_profiles_summary:
            final_profiles_summary[cluster_label] = {
                "name": f"Person {cluster_label + 1}", # Default clustered name
                "frames_seen": []
            }

        # Aggregate frames
        final_profiles_summary[cluster_label]["frames_seen"].extend(profiles[old_pid]["frames_seen"])

        # Logic to determine the best name for the cluster
        # Prioritize known names or names that aren't generic "Person X"
        current_name = final_profiles_summary[cluster_label]["name"]
        original_name = profiles[old_pid].get("name", "")
        if not current_name.startswith("Person ") and original_name.startswith("Person "):
             pass # Keep the existing non-generic name
        elif original_name and not original_name.startswith("Person "):
             final_profiles_summary[cluster_label]["name"] = original_name # Use the identified name


    # Clean up frame lists (sort and unique)
    for label in final_profiles_summary:
        final_profiles_summary[label]["frames_seen"] = sorted(list(set(final_profiles_summary[label]["frames_seen"])))

    print(f"Clustering resulted in {len(final_profiles_summary)} distinct profiles.")
    # Renumber cluster labels to be 1-based sequential IDs for final output
    final_renumbered_profiles = {}
    final_renumbered_assignments = {}
    sorted_labels = sorted(final_profiles_summary.keys())
    label_to_new_id = {label: i + 1 for i, label in enumerate(sorted_labels)}

    for old_pid, cluster_label in new_profile_assignments.items():
        final_renumbered_assignments[old_pid] = label_to_new_id[cluster_label]

    for label in sorted_labels:
        new_id = label_to_new_id[label]
        final_renumbered_profiles[new_id] = final_profiles_summary[label]
        # Update name if it's still generic and can be improved
        if final_renumbered_profiles[new_id]["name"] == f"Person {label + 1}":
            final_renumbered_profiles[new_id]["name"] = f"Person {new_id}"


    return final_renumbered_assignments, final_renumbered_profiles
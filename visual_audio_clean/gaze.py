# gaze_estimation.py
import torch
import config

def load_gaze_model(repo=config.GAZE_MODEL_REPO, model_name=config.GAZE_MODEL_NAME, device=config.device):
    """Loads the Gaze estimation model."""
    print(f"Loading Gaze model {model_name} from {repo}...")
    try:
        model, _ = torch.hub.load(repo, model_name, verbose=False) # Suppress verbose loading output
        model.eval()
        model.to(device)
        print("Gaze model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading Gaze model: {e}")
        raise

def run_gaze_estimation(gaze_model, image_tensors, normalized_bboxes_list, device=config.device):
    """
    Runs gaze estimation on a batch of images and corresponding bounding boxes.

    Args:
        gaze_model: Loaded gaze estimation model.
        image_tensors: List of transformed image tensors.
        normalized_bboxes_list: List where each element is a list of normalized
                                 bboxes [x1, y1, x2, y2] for the corresponding image tensor.
                                 Each bbox list can be empty if no faces were detected.
        device: Computation device ('cuda' or 'cpu').

    Returns:
        Dictionary containing gaze estimation results ('inout', 'heatmap'),
        or None if no valid input.
    """
    if not image_tensors or not any(len(bboxes) > 0 for bboxes in normalized_bboxes_list):
        print("Skipping gaze estimation: No faces detected in any frame.")
        return None

    # Ensure tensors are on the correct device
    img_batch = torch.cat(image_tensors, dim=0).to(device)

    # Prepare input for the gaze model
    # The model expects a list of lists of bounding boxes.
    # Ensure inner lists are correctly formatted (e.g., as NumPy arrays or lists of floats)
    formatted_bboxes = []
    for bboxes in normalized_bboxes_list:
         # Convert to list of lists/arrays if needed, ensure floats
         formatted_bboxes.append([[float(coord) for coord in box] for box in bboxes])

    inp = {"images": img_batch, "bboxes": formatted_bboxes}

    print("Running gaze detection model...")
    with torch.no_grad():
        gaze_out = gaze_model(inp)
    print("Gaze detection complete.")
    return gaze_out
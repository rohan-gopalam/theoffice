# visualization.py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import config # Import config for thresholds etc.

def visualize_all(pil_image, heatmaps, bboxes, inout_scores=None, emotions=None, names=None, profile_ids=None, inout_thresh=config.GAZE_INOUT_THRESHOLD):
    """
    Create a visualization image with bounding boxes, gaze points, IDs, and header text.

    Args:
        pil_image: PIL Image.
        heatmaps: List/Tensor of heatmaps for each face.
        bboxes: List of normalized bounding boxes [x1, y1, x2, y2].
        inout_scores: List/Tensor of In-out scores for gaze (optional).
        emotions: List of emotion strings (optional).
        names: List of face names (optional).
        profile_ids: List of profile IDs (optional).
        inout_thresh: Threshold for considering gaze as "looking at camera".

    Returns:
        An annotated PIL Image.
    """
    colors = ['lime', 'tomato', 'cyan', 'fuchsia', 'yellow', 'orange', 'blueviolet', 'pink']
    width, height = pil_image.size

    # Determine Header Height dynamically
    num_faces = len(bboxes)
    base_font_size = 20
    header_line_height = base_font_size + 5
    title_height = 40 # Approx height for title
    header_height = title_height + (num_faces * header_line_height) + 10 # Add some padding

    output_img = Image.new("RGBA", (width, height + header_height), (255,255,255,255))
    output_img.paste(pil_image, (0, header_height)) # Paste original image below header
    draw = ImageDraw.Draw(output_img)

    # --- Draw Header Background and Separator ---
    draw.rectangle([0, 0, width, header_height], fill=(240,240,240,255)) # Light grey header
    draw.line([(0, header_height), (width, header_height)], fill=(100,100,100,255), width=2)

    header_text_entries = [] # List to store text info for the header

    # --- Draw Face BBoxes and Gaze Lines ---
    for i, norm_box in enumerate(bboxes):
        xmin, ymin, xmax, ymax = norm_box
        # Calculate pixel coordinates relative to the *original image within the output*
        xmin_px = int(xmin * width)
        ymin_px = int(ymin * height) + header_height # Add header offset
        xmax_px = int(xmax * width)
        ymax_px = int(ymax * height) + header_height # Add header offset

        color = colors[i % len(colors)]
        line_width = max(2, int(min(width, height) * 0.005)) # Adjust line width

        # Draw bounding box
        draw.rectangle([xmin_px, ymin_px, xmax_px, ymax_px], outline=color, width=line_width)

        # Prepare info text for this person for the header
        info_list = []
        pid_str = f"ID: {profile_ids[i]}" if profile_ids and i < len(profile_ids) else f"Face {i+1}"
        name_str = f"{names[i]}" if names and i < len(names) else "Unknown"
        info_list.append(f"{pid_str} ({name_str})")

        if emotions is not None and i < len(emotions):
            info_list.append(f"Emotion: {emotions[i]}")

        gaze_info_str = ""
        draw_gaze_line = False
        if inout_scores is not None and i < len(inout_scores):
            score_val = float(inout_scores[i].item() if torch.is_tensor(inout_scores[i]) else inout_scores[i])
            if score_val > inout_thresh:
                gaze_info_str = f"Looking at camera ({score_val:.2f})"
                draw_gaze_line = True
            else:
                gaze_info_str = f"Not looking ({score_val:.2f})"
            info_list.append(gaze_info_str)

        header_text_entries.append((info_list, color))

        # Draw Gaze Point and Line if applicable
        if draw_gaze_line and heatmaps is not None and i < len(heatmaps):
            heat_np = heatmaps[i].detach().cpu().numpy() if torch.is_tensor(heatmaps[i]) else heatmaps[i]
            if heat_np.ndim == 2 and heat_np.size > 0: # Check heatmap validity
                # Find peak in heatmap
                max_idx = np.unravel_index(np.argmax(heat_np), heat_np.shape)
                gaze_y_norm, gaze_x_norm = max_idx[0] / heat_np.shape[0], max_idx[1] / heat_np.shape[1]

                # Convert normalized gaze target to pixel coordinates on the output image
                gaze_x_px = int(gaze_x_norm * width)
                gaze_y_px = int(gaze_y_norm * height) + header_height # Add header offset

                # Calculate face center
                center_x_px = int(((xmin + xmax) / 2) * width)
                center_y_px = int(((ymin + ymax) / 2) * height) + header_height

                radius = max(3, int(min(width, height) * 0.008)) # Gaze point radius
                # Draw gaze target point
                draw.ellipse([(gaze_x_px - radius, gaze_y_px - radius), (gaze_x_px + radius, gaze_y_px + radius)], fill=color, outline='black', width=1)
                # Draw line from face center to gaze point
                draw.line([(center_x_px, center_y_px), (gaze_x_px, gaze_y_px)], fill=color, width=line_width)
            else:
                 print(f"Warning: Invalid heatmap shape {heat_np.shape} for face {i}")

    # --- Draw Header Text ---
    try:
        font = ImageFont.truetype("arial.ttf", base_font_size) # Common font
    except IOError:
        font = ImageFont.load_default() # Fallback

    y_offset = title_height # Start drawing text below the title area

    for info, col in header_text_entries:
        box_size = base_font_size * 0.8 # Size of the color swatch
        draw.rectangle([10, y_offset, 10 + box_size, y_offset + box_size], fill=col)
        draw.text((10 + box_size + 5, y_offset), " | ".join(info), fill="black", font=font)
        y_offset += header_line_height

    # --- Draw Title ---
    title = f"Frame Analysis ({len(bboxes)} face(s) detected)"
    title_font_size = 28
    try:
        title_font = ImageFont.truetype("arialbd.ttf", title_font_size) # Bold Arial
    except IOError:
        try:
            title_font = ImageFont.truetype("arial.ttf", title_font_size)
        except IOError:
            title_font = ImageFont.load_default()

    # Calculate text width for centering using TextualExtents if available (more accurate)
    if hasattr(title_font, 'getbbox'): # PIL 9.0.0+
        bbox = title_font.getbbox(title)
        title_width = bbox[2] - bbox[0]
        # title_height_ = bbox[3] - bbox[1] # If needed
    elif hasattr(title_font, 'getsize'): # Older PIL/Pillow
         title_width, _ = title_font.getsize(title)
    else: # Fallback approximation
         title_width = len(title) * title_font_size * 0.6

    title_x = max(10, (width - title_width) / 2) # Ensure it doesn't go off-left
    draw.text((title_x, 5), title, fill="black", font=title_font) # Position title near top

    return output_img
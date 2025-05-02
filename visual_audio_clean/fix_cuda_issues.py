import torch
import torchvision

# Original NMS function from torchvision
original_nms = torchvision.ops.nms

# Create a wrapper function that ensures tensors are on the same device
def safe_nms(boxes, scores, iou_threshold):
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Make sure both tensors are on the same device
        if boxes.is_cuda and not scores.is_cuda:
            scores = scores.cuda()
        elif not boxes.is_cuda and scores.is_cuda:
            boxes = boxes.cuda()
    else:
        # If CUDA not available, ensure both are on CPU
        boxes = boxes.cpu()
        scores = scores.cpu()
    
    # Call the original NMS function with properly aligned devices
    return original_nms(boxes, scores, iou_threshold)

# Monkey patch the NMS function
torchvision.ops.nms = safe_nms
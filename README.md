# install 
`pip install ultralytics opencv-python pydub numpy scipy `

`pip install torch torchvision torchaudio # Dependencies for ultralytics/YOLO`

# Example command
if u have m1/2/3/4 mac chip use: `python spacebar.py "videos/ween.mp4" --output_json "labels.json" --yolo_model "yolov8n.pt" --device "mps"`

otherwise: `python spacebar.py "videos/ween.mp4" --output_json "labels.json" --yolo_model "yolov8n.pt"`

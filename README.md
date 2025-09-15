# YOLOv8 Object Detection on Video
This project performs real-time object detection on a video using the YOLOv8 model with GPU acceleration. The processed video with bounding boxes and confidence scores is saved as an output file.

![Result](https://github.com/ayhmdalila/traffic-detection/blob/main/results.gif)

## Features

- Detect objects in a video using YOLOv8.
- GPU acceleration with PyTorch.
- Draw bounding boxes with labels and confidence scores.
- Save processed video.
- Optional real-time display of processed frames.



## Requirements

- Python 3.12
- GPU with CUDA support for faster inference.
- Packages listed in `requirements.txt`.



## Setup

1. **Clone the repository**:

```bash
git clone <repository_url>
cd <repository_directory>
```
2. **Install dependencies**:
```bash
pip install -r requirements.txt
```
## Usage

Run the main script:

```bash
python main.py
```

### Run Docker container:

```bash
docker run --gpus all -v $(pwd):/app yolo-video-detector
```

- Make sure to mount the project directory to `/app` in the container.
- Ensure your GPU drivers and Docker NVIDIA runtime are properly set up.



## Configuration

- **Frame size**: Currently set to `640x360`.
- **Frame skipping**: Can skip frames by adjusting `frame_skip`.
- **YOLO confidence threshold**: Set via `conf` parameter (default 0.4).
- **IOU threshold**: Set via `iou` parameter (default 0.5).



## Notes

- GPU is required for optimal performance.
- For CPU-only mode, remove `.to("cuda")` in `main.py`.
- Supports `.mp4`, `.mkv`, and other common video formats.



## License

Open source

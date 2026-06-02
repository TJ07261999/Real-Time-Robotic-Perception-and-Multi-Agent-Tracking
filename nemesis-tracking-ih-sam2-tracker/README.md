# Nemesis Vision

<p align="center">
  <img src="assets/fight2.gif" alt="animated" />
</p>

## Getting Started

- Python 3.10+
- [SAM 2](https://github.com/facebookresearch/sam2)

## Setup

1. **Clone the repository:**

   ```bash
   git clone git@github.com:AdvancedRoboticCombat/nemesis-tracking.git
   cd nemesis-tracking
   ```

2. **Create and activate environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   # Install PyTorch (Check pytorch.org for your specific system command)
   pip install torch torchvision torchaudio

   # Install other dependencies
   pip install matplotlib opencv-python supervision
   pip install git+https://github.com/facebookresearch/sam2.git
   ```

4. **Prepare Model & Configs:**
   The script expects the following directory structure. You must download the model checkpoint and config:

   ```bash
   mkdir -p ckpts configs/sam2.1

   # Download the tiny model checkpoint
   curl -o ckpts/sam2.1_hiera_tiny.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_tiny.pt

   # Download the matching config file
   curl -o configs/sam2.1/sam2.1_hiera_t.yaml https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_t.yaml
   ```

## Usage

1. **Configure input:**
   Edit `main.py` to point to your input video file:

   ```python
   VIDEO_PATH = "data/fight5-sample.mp4"
   OUT_VIDEO = "out/fight5-sample.mp4"
   ```

2. **Run the tracker:**

   ```bash
   python main.py
   ```

3. **Interactive Controls:**
   A window will open showing the first frame of the video.
   - **Left Click**: Add a positive point (green) to the object you want to track.
   - **Right Click**: Add a negative point (red) to exclude an area.
   - **`n` key**: Finish adding points for the current object and start the next one (e.g., Nemesis -> Opponent).
   - **`q` key**: Finish setup and begin tracking processing.

## Output

The processed video with tracking overlays will be saved to the path specified in `OUT_VIDEO`.

## Future Work

![mot-pipeline](assets/mot_pipeline.png)

# Advanced Document Scanner Pipeline

Computer Vision pipeline: **Enhance → Segment → Clean → Detect → Decide**

## Quick Start

```bash
pip install -r requirements.txt

# Single image
python src/main.py input.jpg --enhancement clahe --detection auto

# Method comparison (all combinations)
python src/main.py input.jpg --compare

# Gradio UI
python src/app.py

# Video processing
python src/video.py input.mp4

# Generate test dataset
python samples/generate_test.py
```

## Pipeline Stages

1. **Enhancement** - CLAHE, Gamma correction, Histogram Equalization, Denoising
2. **Segmentation** - Canny edge detection with automatic thresholds
3. **Cleaning** - Morphological operations (dilation, closing, opening)
4. **Detection** - Contour / Hough / Edge-analysis document detection
5. **Decision** - Quality scoring, blur/contrast analysis, pass/fail

## Output

- `output/visualization/` - All pipeline stage images
- `output/visualization/pipeline_summary.png` - Side-by-side comparison
- `output/visualization/decision.txt` - Quality metrics and final decision
- `output/comparison/` - Method benchmarking results

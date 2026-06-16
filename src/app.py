import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tempfile
from pipeline import run_pipeline
from config import CONFIG

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Run: pip install gradio")
    print("Fallback: use main.py CLI instead.")
    sys.exit(1)


def process_image(image, enhancement, detection, upscale):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name
        image.save(temp_path)

    result = run_pipeline(
        temp_path,
        enhancement=enhancement,
        detection_strategy=detection,
        upscale=upscale,
        output_dir=tempfile.mkdtemp(),
        save_images=False,
    )

    os.unlink(temp_path)

    if not result.success:
        return (
            None, None, None, None, None,
            f"FAILED: {result.decision_reason}"
        )

    stages = result.stage_images
    display = {}

    if "original" in stages:
        display["Original"] = stages["original"][:, :, ::-1]
    if "enhanced" in stages:
        display["Enhanced"] = stages["enhanced"][:, :, ::-1]
    if "edges" in stages:
        display["Edges"] = stages["edges"]
    if "detected_document" in stages:
        display["Corrected"] = stages["detected_document"][:, :, ::-1]
    if "thresholded" in stages:
        display["Thresholded"] = stages["thresholded"]
    if "cleaned_mask" in stages:
        display["Cleaned"] = stages["cleaned_mask"]

    metrics_text = "\n".join(
        f"  {k.replace('_', ' ').title()}: {v:.2f}"
        for k, v in result.metrics.items()
    )

    decision_text = (
        f"Grade: {result.decision_grade}\n"
        f"Reason: {result.decision_reason}\n"
        f"Quality Score: {result.quality_score:.1f}%\n"
        f"Detection: {result.detection_strategy_used}\n"
        f"Enhancement: {result.enhancement_method_used}\n\n"
        f"--- Metrics ---\n{metrics_text}"
    )

    return (
        display.get("Original"),
        display.get("Enhanced"),
        display.get("Edges"),
        display.get("Corrected"),
        display.get("Thresholded"),
        decision_text,
    )


def build_ui():
    with gr.Blocks(title="Document Scanner Pipeline", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Document Scanner + Enhancer
            Computer Vision pipeline: **Enhance → Segment → Clean → Detect → Decide**
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Input Image")
                enhancement = gr.Dropdown(
                    choices=["clahe", "gamma", "hist_eq", "denoise", "none"],
                    value=CONFIG.default_enhancement,
                    label="Enhancement Method"
                )
                detection = gr.Dropdown(
                    choices=["contour", "hough", "edge", "auto"],
                    value=CONFIG.default_detection,
                    label="Detection Strategy"
                )
                upscale = gr.Slider(1.0, 4.0, value=2.0, step=0.5,
                                    label="Upscale Factor")
                run_btn = gr.Button("Run Pipeline", variant="primary")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Original"):
                        orig_out = gr.Image(label="Original")
                    with gr.TabItem("Enhanced"):
                        enh_out = gr.Image(label="Enhanced")
                    with gr.TabItem("Edges"):
                        edges_out = gr.Image(label="Edge Detection")
                    with gr.TabItem("Corrected"):
                        corr_out = gr.Image(label="Perspective Corrected")
                    with gr.TabItem("Thresholded"):
                        thresh_out = gr.Image(label="Thresholded")
                    with gr.TabItem("Decision"):
                        decision_out = gr.Textbox(label="Decision",
                                                  lines=15)

        run_btn.click(
            fn=process_image,
            inputs=[image_input, enhancement, detection, upscale],
            outputs=[orig_out, enh_out, edges_out, corr_out,
                     thresh_out, decision_out],
        )

        gr.Markdown(
            """
            ### Pipeline Stages
            1. **Enhancement**: CLAHE, Gamma, Histogram Equalization, or Denoising
            2. **Segmentation**: Canny edge detection with automatic thresholds
            3. **Cleaning**: Morphological operations (dilation, closing, opening)
            4. **Detection**: Contour / Hough / Edge-analysis based document detection
            5. **Decision**: Quality scoring + Pass/Fail grade

            ### Bonus Features
            - CLI with `--compare` flag for method benchmarking
            - Video processing support
            - Comprehensive visualization outputs
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False)

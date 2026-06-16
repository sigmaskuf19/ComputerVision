import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


def prepare_save_dir(output_dir: str = "output") -> str:
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    return vis_dir


def save_stage_image(img: np.ndarray, name: str,
                     output_dir: str = "output") -> str:
    vis_dir = prepare_save_dir(output_dir)
    path = os.path.join(vis_dir, f"{name}.png")
    cv2.imwrite(path, img)
    return path


def save_decision_output(decision_text: str, output_dir: str = "output") -> str:
    vis_dir = prepare_save_dir(output_dir)
    path = os.path.join(vis_dir, "decision.txt")
    with open(path, "w") as f:
        f.write(decision_text)
    return path


def create_pipeline_summary(stage_images: Dict[str, np.ndarray],
                            output_dir: str = "output",
                            show: bool = False) -> str:
    vis_dir = prepare_save_dir(output_dir)

    n_stages = len(stage_images)
    cols = min(4, n_stages)
    rows = (n_stages + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i, (name, img) in enumerate(stage_images.items()):
        if i < len(axes):
            ax = axes[i]
            if len(img.shape) == 2 or img.shape[2] == 1:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(name.replace("_", " ").title(), fontsize=10)
            ax.axis("off")

    for i in range(n_stages, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    summary_path = os.path.join(vis_dir, "pipeline_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()

    return summary_path


def create_detailed_report(stage_images: Dict[str, np.ndarray],
                           metrics: dict,
                           decision: Tuple[str, str],
                           output_dir: str = "output") -> Dict[str, str]:
    vis_dir = prepare_save_dir(output_dir)
    paths = {}

    for name, img in stage_images.items():
        path = save_stage_image(img, name, output_dir)
        paths[name] = path

    summary = create_pipeline_summary(stage_images, output_dir)
    paths["summary"] = summary

    grade, reason = decision
    lines = [
        "=" * 50,
        "DOCUMENT SCANNER - FINAL DECISION",
        "=" * 50,
        f"Grade: {grade}",
        f"Reason: {reason}",
        "",
        "--- Quality Metrics ---",
    ]
    for k, v in metrics.items():
        lines.append(f"  {k.replace('_', ' ').title()}: {v:.2f}")

    decision_text = "\n".join(lines)

    with open(os.path.join(vis_dir, "decision.txt"), "w") as f:
        f.write(decision_text)

    with open(os.path.join(vis_dir, "report.txt"), "w") as f:
        f.write(decision_text)

    paths["decision"] = os.path.join(vis_dir, "decision.txt")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.text(0.1, 0.9, decision_text, fontsize=11, family="monospace",
            verticalalignment="top", transform=ax.transAxes)
    plt.tight_layout()
    decision_img = os.path.join(vis_dir, "decision_overlay.png")
    plt.savefig(decision_img, dpi=150, bbox_inches="tight")
    plt.close()
    paths["decision_img"] = decision_img

    return paths


def visualize_corner_detection(img: np.ndarray,
                               corners: np.ndarray,
                               output_dir: str = "output",
                               filename: str = "corner_detection") -> str:
    vis = img.copy()
    for i, pt in enumerate(corners):
        pt_int = tuple(pt.astype(int))
        cv2.circle(vis, pt_int, 12, (0, 255, 0), -1)
        labels = ["TL", "TR", "BR", "BL"]
        cv2.putText(vis, labels[i], (pt_int[0] + 14, pt_int[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.polylines(vis, [corners.astype(np.int32)], True, (0, 255, 180), 2)

    vis_dir = prepare_save_dir(output_dir)
    path = os.path.join(vis_dir, f"{filename}.png")
    cv2.imwrite(path, vis)
    return path

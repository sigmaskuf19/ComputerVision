import cv2
import os
from typing import Optional
from pipeline import run_pipeline


def process_video(video_path: str,
                  enhancement: str = "clahe",
                  detection_strategy: str = "auto",
                  output_dir: str = "output",
                  frame_skip: int = 5,
                  max_frames: Optional[int] = None) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}"}

    os.makedirs(output_dir, exist_ok=True)
    results = []
    frame_count = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.png")
        cv2.imwrite(frame_path, frame)

        result = run_pipeline(
            frame_path,
            enhancement=enhancement,
            detection_strategy=detection_strategy,
            output_dir=os.path.join(output_dir, f"frames/frame_{frame_count:06d}"),
            save_images=False,
        )

        results.append({
            "frame": frame_count,
            "success": result.success,
            "quality_score": result.quality_score,
            "decision_grade": result.decision_grade,
        })

        processed += 1
        if max_frames and processed >= max_frames:
            break

        frame_count += 1

    cap.release()

    summary_path = os.path.join(output_dir, "video_results.txt")
    with open(summary_path, "w") as f:
        f.write(f"Video: {video_path}\n")
        f.write(f"Frames processed: {processed}\n")
        successes = sum(1 for r in results if r["success"])
        f.write(f"Successful detections: {successes}/{processed}\n")
        if successes:
            avg_score = sum(r["quality_score"] for r in results
                           if r["success"]) / successes
            f.write(f"Average quality score: {avg_score:.1f}%\n")
        f.write("\nPer-frame results:\n")
        for r in results:
            f.write(f"  Frame {r['frame']}: {r['decision_grade']} "
                    f"(score: {r['quality_score']:.1f})\n")

    return {
        "frames_processed": processed,
        "successful": successes,
        "total": len(results),
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process video through document scanner")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--enhancement", default="clahe")
    parser.add_argument("--detection", default="auto")
    parser.add_argument("--output", "-o", default="output/video")
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    result = process_video(
        args.video_path,
        enhancement=args.enhancement,
        detection_strategy=args.detection,
        output_dir=args.output,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
    )

    print(f"Processed {result.get('frames_processed', 0)} frames")
    if "error" in result:
        print(f"Error: {result['error']}")

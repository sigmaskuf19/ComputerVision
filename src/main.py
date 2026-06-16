import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import run_pipeline, run_comparison
from visualization import create_pipeline_summary
from config import CONFIG


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Document Scanner Pipeline"
    )
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--enhancement", "-e", default=CONFIG.default_enhancement,
                        choices=["clahe", "gamma", "hist_eq", "denoise", "none"],
                        help="Enhancement method")
    parser.add_argument("--detection", "-d", default=CONFIG.default_detection,
                        choices=["contour", "hough", "edge", "auto"],
                        help="Document detection strategy")
    parser.add_argument("--upscale", "-u", type=float, default=CONFIG.upscale_factor,
                        help="Upscale factor")
    parser.add_argument("--output", "-o", default=CONFIG.output_dir,
                        help="Output directory")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save intermediate images")
    parser.add_argument("--compare", action="store_true",
                        help="Run all method combinations and compare")
    parser.add_argument("--summary", action="store_true",
                        help="Show pipeline summary visualization")

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: file not found: {args.image_path}")
        sys.exit(1)

    if args.compare:
        print("Running full method comparison...")
        results = run_comparison(args.image_path, args.output)

        print("\n" + "=" * 60)
        print("METHOD COMPARISON RESULTS")
        print("=" * 60)
        best = None
        best_score = -1
        for combo, res in sorted(results.items()):
            status = "OK" if res.success else "FAIL"
            score = f"{res.quality_score:.1f}%" if res.success else "N/A"
            grade = res.decision_grade if res.success else "FAIL"
            det = res.detection_strategy_used if res.success else "-"
            print(f"  {combo:25s} | {status:4s} | Score: {score:6s} | {grade:22s} | det: {det}")
            if res.success and res.quality_score > best_score:
                best_score = res.quality_score
                best = combo

        if best:
            print(f"\nBest combination: {best} (score: {best_score:.1f}%)")
        return

    result = run_pipeline(
        args.image_path,
        enhancement=args.enhancement,
        detection_strategy=args.detection,
        upscale=args.upscale,
        output_dir=args.output,
        save_images=not args.no_save,
    )

    grade, reason = result.decision
    print(f"\nPipeline: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Detection strategy used: {result.detection_strategy_used}")
    print(f"Enhancement method used: {result.enhancement_method_used}")
    print(f"Quality score: {result.quality_score:.1f}%")
    print(f"Decision grade: {grade}")
    print(f"Reason: {reason}")

    if result.metrics:
        print("\nQuality Metrics:")
        for k, v in result.metrics.items():
            print(f"  {k.replace('_', ' ').title()}: {v:.2f}")

    if result.success and args.summary:
        summary_path = create_pipeline_summary(
            result.stage_images, args.output
        )
        print(f"\nPipeline summary saved: {summary_path}")

    if not args.no_save:
        print(f"\nAll outputs saved to: {args.output}/visualization/")

    if not result.success and result.error:
        print(f"\nError: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import run_comparison


def generate_comparison_report(image_path: str,
                                output_dir: str = "output/comparison"):
    print(f"Running comparison on: {image_path}")
    results = run_comparison(image_path, output_dir)

    rows = []
    for combo, res in sorted(results.items()):
        rows.append({
            "combo": combo,
            "success": res.success,
            "quality_score": round(res.quality_score, 1) if res.success else None,
            "grade": res.decision_grade if res.success else "FAIL",
            "detection": res.detection_strategy_used if res.success else "-",
            "error": res.error,
        })

    report_path = os.path.join(output_dir, "comparison_results.json")
    with open(report_path, "w") as f:
        json.dump(rows, f, indent=2)

    print("\n" + "=" * 70)
    print(f"{'Method Combination':30s} {'Status':6s} {'Score':8s} {'Grade':25s}")
    print("=" * 70)
    for r in rows:
        score = f"{r['quality_score']:.1f}%" if r["quality_score"] else "N/A"
        print(f"{r['combo']:30s} {'OK' if r['success'] else 'FAIL':6s} "
              f"{score:8s} {r['grade']:25s}")

    best = max([r for r in rows if r["success"]],
               key=lambda r: r["quality_score"], default=None)
    if best:
        print(f"\nBest: {best['combo']} ({best['quality_score']:.1f}%)")

    print(f"\nFull report: {report_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to test image")
    parser.add_argument("--output", "-o", default="output/comparison")
    args = parser.parse_args()
    generate_comparison_report(args.image_path, args.output)

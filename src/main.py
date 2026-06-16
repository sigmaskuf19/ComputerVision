import argparse
import cv2
from utils import show2, scan_document


def main():
    parser = argparse.ArgumentParser(description="Document scanner pipeline")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--upscale", type=float, default=2.0, help="Upscale factor (default: 2.0)")
    parser.add_argument("--debug", action="store_true", help="Show debug visualizations")
    parser.add_argument("--output", "-o", help="Output path prefix for saved results")
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.image_path}")

    if args.upscale != 1.0:
        image = cv2.resize(
            image, None,
            fx=args.upscale, fy=args.upscale,
            interpolation=cv2.INTER_CUBIC
        )

    color, clean = scan_document(image, debug=args.debug)
    if color is None:
        print("Document detection failed.")
        return

    print(f"Output: {color.shape[1]}x{color.shape[0]}")

    if args.debug:
        show2(color, "Color scan", clean, "Clean scan")

    if args.output:
        cv2.imwrite(f"{args.output}_color.jpg", color)
        cv2.imwrite(f"{args.output}_clean.jpg", clean)
        print(f"Saved: {args.output}_color.jpg, {args.output}_clean.jpg")


if __name__ == "__main__":
    main()

import cv2
import numpy as np
import os


def generate_test_document(filename: str, text_lines: int = 15,
                           width: int = 1200, height: int = 1600):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    margin = 80
    y = margin

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    line_height = 40

    title = "Sample Document for OCR Pipeline Testing"
    cv2.putText(img, title, (margin, y), font, 0.9, (0, 0, 0), 2)
    y += int(line_height * 1.8)

    cv2.line(img, (margin, y), (width - margin, y), (0, 0, 0), 1)
    y += line_height

    for i in range(text_lines):
        words = np.random.randint(5, 12)
        text = " ".join(
            f"word{np.random.randint(100, 999)}" for _ in range(words)
        )
        prefix = f"{i + 1}. "
        cv2.putText(img, prefix + text, (margin, y), font,
                    font_scale, (0, 0, 0), thickness)
        y += line_height

    y += line_height
    cv2.putText(img, "END OF DOCUMENT", (margin, y), font,
                0.7, (100, 100, 100), 1)

    cv2.imwrite(filename, img)
    print(f"Generated: {filename} ({width}x{height})")


def apply_perspective_distortion(img: np.ndarray,
                                 skew: float = 0.15) -> np.ndarray:
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [int(w * skew * 0.3), int(h * skew * 0.2)],
        [int(w * (1 - skew * 0.5)), int(h * skew * 0.1)],
        [int(w * (1 - skew * 0.3)), int(h * (1 - skew * 0.2))],
        [int(w * skew * 0.4), int(h * (1 - skew * 0.1))],
    ])
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, H, (w, h))


def generate_dataset(output_dir: str = "samples", count: int = 5):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(count):
        clean_path = os.path.join(output_dir, f"doc_{i:02d}_clean.png")
        generate_test_document(clean_path)

        img = cv2.imread(clean_path)

        distorted = apply_perspective_distortion(img, skew=0.1 + 0.05 * i)
        distorted_path = os.path.join(output_dir, f"doc_{i:02d}_distorted.png")
        cv2.imwrite(distorted_path, distorted)
        print(f"Generated: {distorted_path}")

        noisy = cv2.GaussianBlur(img, (5, 5), 1.5)
        noisy_path = os.path.join(output_dir, f"doc_{i:02d}_blurred.png")
        cv2.imwrite(noisy_path, noisy)
        print(f"Generated: {noisy_path}")

    print(f"\nDataset generated in '{output_dir}/' with {count} documents "
          f"(3 variants each = {count * 3} images)")


if __name__ == "__main__":
    generate_dataset()

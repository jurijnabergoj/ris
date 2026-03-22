from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent


def crop_left_half(path):
    img = Image.open(path)
    img.crop((0, 0, img.width // 2, img.height)).save(path)


if __name__ == "__main__":
    # Crop training images that contain two side-by-side petri dishes to the left half
    train_dir = PROJECT_ROOT / "data" / "ris2026-krog1-ucni-test"
    for f in ["75c8bd04.png", "3075a94c.png", "8501bff5.png", "2283929d.png"]:
        crop_left_half(train_dir / f)
        print(f"Cropped training image: {f}")

    # Crop one test image where the dish is on the left and the right half is empty.
    test_dir = PROJECT_ROOT / "data" / "ris2026-krog1-testni-test"
    crop_left_half(test_dir / "b40ccdbd.png")
    print("Cropped test image: b40ccdbd.png")

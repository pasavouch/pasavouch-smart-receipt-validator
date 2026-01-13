import cv2
from skimage.metrics import structural_similarity as ssim
import sys
import os

TEMPLATE = "template_smart_receipt_v1.jpg"
THRESHOLD = 0.90

def validate(image_path):
    if not os.path.exists(image_path):
        print("ERROR: image file not found")
        return

    ref = cv2.imread(TEMPLATE, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if ref is None or img is None:
        print("ERROR: cannot read image")
        return

    img = cv2.resize(img, (ref.shape[1], ref.shape[0]))
    score, _ = ssim(ref, img, full=True)

    if score >= THRESHOLD:
        print(f"VALID receipt (Similarity: {score:.2f})")
    else:
        print(f"REJECTED receipt (Similarity: {score:.2f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Gamit: python validator.py uploaded_receipt.jpg")
    else:
        validate(sys.argv[1])
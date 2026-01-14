from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

app = Flask(__name__)
CORS(app)

# Base directory of the server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Reference template image used for format comparison
TEMPLATE_PATH = os.path.join(BASE_DIR, "template_smart_receipt_v1.jpg")

# Load template image once into memory
REF_IMG = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)


# Validation configuration values
THRESHOLD = 0.80      # Minimum SSIM score to pass format validation
DIFF_LIMIT = 30       # Maximum allowed pixel difference
EDGE_LIMIT = 15       # Edge density limit for detecting overlays
ASPECT_TOL = 0.12     # Aspect ratio tolerance


# Function for Image Format Checker
# This endpoint validates if the uploaded image follows
# the official Smart receipt format before OCR processing
@app.route("/validate-format", methods=["POST"])
def validate_format():

    # Check if an image file is included in the request
    if "image" not in request.files:
        return jsonify({"ok": False, "reason": "NO_IMAGE"})

    try:
        # Read uploaded image bytes
        file = request.files["image"]
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

        # Validate image decoding
        if img is None or REF_IMG is None:
            return jsonify({"ok": False, "reason": "IMAGE_READ_ERROR"})

        # Compare aspect ratio with reference template
        h_ref, w_ref = REF_IMG.shape
        ratio_ref = w_ref / h_ref
        ratio_img = img.shape[1] / img.shape[0]

        if abs(ratio_ref - ratio_img) > ASPECT_TOL:
            return jsonify({"ok": False, "reason": "ASPECT_RATIO_MISMATCH"})

        # Resize image to match template dimensions
        img_resized = cv2.resize(img, (w_ref, h_ref))

        # Pre-process images to reduce noise
        img_proc = cv2.GaussianBlur(img_resized, (5, 5), 0)
        ref_proc = cv2.GaussianBlur(REF_IMG, (5, 5), 0)

        # Crop the main body area for comparison
        y1, y2 = int(h_ref * 0.27), int(h_ref * 0.77)
        x1, x2 = int(w_ref * 0.22), int(w_ref * 0.78)

        ref_crop = ref_proc[y1:y2, x1:x2]
        img_crop = img_proc[y1:y2, x1:x2]

        # Detect overlay elements such as watermarks or edits
        wm_y1, wm_y2 = int(h_ref * 0.36), int(h_ref * 0.56)
        wm_x1, wm_x2 = int(w_ref * 0.32), int(w_ref * 0.68)

        edges = cv2.Canny(
            img_resized[wm_y1:wm_y2, wm_x1:wm_x2],
            80,
            200
        )

        if edges.mean() > EDGE_LIMIT:
            return jsonify({"ok": False, "reason": "OVERLAY_DETECTED"})

        # Pixel difference check against reference template
        if cv2.absdiff(ref_crop, img_crop).mean() > DIFF_LIMIT:
            return jsonify({"ok": False, "reason": "TEMPLATE_DIFF_TOO_HIGH"})

        # Structural similarity comparison (SSIM)
        ref_edge = cv2.Canny(ref_crop, 80, 200)
        img_edge = cv2.Canny(img_crop, 80, 200)

        score, _ = ssim(ref_edge, img_edge, full=True)
        score = round(score, 2)

        if score >= THRESHOLD:
            return jsonify({"ok": True, "similarity": score})

        return jsonify({
            "ok": False,
            "reason": "FORMAT_MISMATCH",
            "similarity": score
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "reason": "SYSTEM_ERROR",
            "msg": str(e)
        })


# Start the Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

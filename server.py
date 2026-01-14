from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

app = Flask(__name__)
CORS(app)

# =========================
# BASE PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Reference template image
TEMPLATE_PATH = os.path.join(BASE_DIR, "template_smart_receipt_v1.jpg")

# Load template once
REF_IMG = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)

if REF_IMG is None:
    raise RuntimeError("Template image not found or failed to load")

# =========================
# FORMAT VALIDATION CONFIG
# =========================
THRESHOLD = 0.80
DIFF_LIMIT = 30
EDGE_LIMIT = 15
ASPECT_TOL = 0.12


# =========================
# FORMAT VALIDATION ENDPOINT
# =========================
@app.route("/validate-format", methods=["POST"])
def validate_format():

    if "image" not in request.files:
        return jsonify({"ok": False, "reason": "NO_IMAGE"})

    try:
        file = request.files["image"]
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return jsonify({"ok": False, "reason": "IMAGE_READ_ERROR"})

        # Aspect ratio check
        h_ref, w_ref = REF_IMG.shape
        ratio_ref = w_ref / h_ref
        ratio_img = img.shape[1] / img.shape[0]

        if abs(ratio_ref - ratio_img) > ASPECT_TOL:
            return jsonify({"ok": False, "reason": "ASPECT_RATIO_MISMATCH"})

        # Resize to template
        img_resized = cv2.resize(img, (w_ref, h_ref))

        # Blur
        img_proc = cv2.GaussianBlur(img_resized, (5, 5), 0)
        ref_proc = cv2.GaussianBlur(REF_IMG, (5, 5), 0)

        # Crop region (content area)
        y1, y2 = int(h_ref * 0.27), int(h_ref * 0.77)
        x1, x2 = int(w_ref * 0.22), int(w_ref * 0.78)

        ref_crop = ref_proc[y1:y2, x1:x2]
        img_crop = img_proc[y1:y2, x1:x2]

        # Watermark / overlay detection
        wm_y1, wm_y2 = int(h_ref * 0.36), int(h_ref * 0.56)
        wm_x1, wm_x2 = int(w_ref * 0.32), int(w_ref * 0.68)

        edges = cv2.Canny(
            img_resized[wm_y1:wm_y2, wm_x1:wm_x2],
            80,
            200
        )

        if edges.mean() > EDGE_LIMIT:
            return jsonify({"ok": False, "reason": "OVERLAY_DETECTED"})

        # Difference check
        if cv2.absdiff(ref_crop, img_crop).mean() > DIFF_LIMIT:
            return jsonify({"ok": False, "reason": "TEMPLATE_DIFF_TOO_HIGH"})

        # Structural similarity
        ref_edge = cv2.Canny(ref_crop, 80, 200)
        img_edge = cv2.Canny(img_crop, 80, 200)

        score, _ = ssim(ref_edge, img_edge, full=True)
        score = round(score, 2)

        if score >= THRESHOLD:
            return jsonify({
                "ok": True,
                "similarity": score
            })

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


# =========================
# START SERVER
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

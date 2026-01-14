from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# TEMPLATE
TEMPLATE_PATH = os.path.join(BASE_DIR, "template_smart_receipt_v1.jpg")

# UPLOAD FOLDER (organized)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads", "receipts")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load template once to memory
REF_IMG = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)

# Config constants
THRESHOLD = 0.80
DIFF_LIMIT = 30
EDGE_LIMIT = 15
ASPECT_TOL = 0.12


@app.route("/validate-format", methods=["POST"])
def validate_format():
    if "image" not in request.files:
        return jsonify({"ok": False, "reason": "NO_IMAGE"})

    try:
        file = request.files["image"]

        # ===== SAVE UPLOADED IMAGE (WITH DATE) =====
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"{timestamp}_receipt.jpg"
        save_path = os.path.join(UPLOAD_DIR, filename)
        file.save(save_path)

        # ===== LOAD IMAGE FOR VALIDATION =====
        img = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)

        if img is None or REF_IMG is None:
            return jsonify({"ok": False, "reason": "IMAGE_READ_ERROR"})

        # ===== ASPECT RATIO CHECK =====
        h_ref, w_ref = REF_IMG.shape
        ratio_ref = w_ref / h_ref
        ratio_img = img.shape[1] / img.shape[0]

        if abs(ratio_ref - ratio_img) > ASPECT_TOL:
            return jsonify({"ok": False, "reason": "ASPECT_RATIO_MISMATCH"})

        img_resized = cv2.resize(img, (w_ref, h_ref))

        # ===== PRE-PROCESSING =====
        img_proc = cv2.GaussianBlur(img_resized, (5, 5), 0)
        ref_proc = cv2.GaussianBlur(REF_IMG, (5, 5), 0)

        # ===== BODY CROP =====
        y1, y2 = int(h_ref * 0.27), int(h_ref * 0.77)
        x1, x2 = int(w_ref * 0.22), int(w_ref * 0.78)

        ref_crop = ref_proc[y1:y2, x1:x2]
        img_crop = img_proc[y1:y2, x1:x2]

        # ===== OVERLAY CHECK =====
        wm_y1, wm_y2 = int(h_ref * 0.36), int(h_ref * 0.56)
        wm_x1, wm_x2 = int(w_ref * 0.32), int(w_ref * 0.68)

        edges = cv2.Canny(
            img_resized[wm_y1:wm_y2, wm_x1:wm_x2],
            80,
            200
        )

        if edges.mean() > EDGE_LIMIT:
            return jsonify({"ok": False, "reason": "OVERLAY_DETECTED"})

        # ===== DIFF CHECK =====
        if cv2.absdiff(ref_crop, img_crop).mean() > DIFF_LIMIT:
            return jsonify({"ok": False, "reason": "TEMPLATE_DIFF_TOO_HIGH"})

        # ===== SSIM CHECK =====
        ref_edge = cv2.Canny(ref_crop, 80, 200)
        img_edge = cv2.Canny(img_crop, 80, 200)

        score, _ = ssim(ref_edge, img_edge, full=True)
        score = round(score, 2)

        if score >= THRESHOLD:
            return jsonify({
                "ok": True,
                "similarity": score,
                "file": filename
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

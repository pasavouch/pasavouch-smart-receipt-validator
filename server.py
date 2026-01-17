from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

app = Flask(__name__)
CORS(app)

# base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# template image
TEMPLATE_PATH = os.path.join(BASE_DIR, "template_smart_receipt_v1.jpg")

# load template
REF_IMG = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
if REF_IMG is None:
    raise RuntimeError("Template image not found")

# config
THRESHOLD = 0.80
DIFF_LIMIT = 30
EDGE_LIMIT = 15
ASPECT_TOL = 0.12


@app.route("/validate-format", methods=["POST"])
def validate_format():

    if "image" not in request.files:
        return jsonify({"ok": False, "reason": "NO_IMAGE"})

    try:
        # read image
        file = request.files["image"]
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return jsonify({"ok": False, "reason": "IMAGE_READ_ERROR"})

        # content area for quality checks
        h, w = img.shape
        content = img[
            int(h * 0.25):int(h * 0.80),
            int(w * 0.20):int(w * 0.80)
        ]

        # brightness check
        mean_brightness = content.mean()
        if mean_brightness < 60:
            return jsonify({"ok": False, "reason": "IMAGE_TOO_DARK"})
        if mean_brightness > 220:
            return jsonify({"ok": False, "reason": "IMAGE_TOO_BRIGHT"})

        # blur check
        blur_score = cv2.Laplacian(content, cv2.CV_64F).var()
        if blur_score < 80:
            return jsonify({"ok": False, "reason": "IMAGE_TOO_BLURRY"})

        # aspect ratio check
        h_ref, w_ref = REF_IMG.shape
        ratio_ref = w_ref / h_ref
        ratio_img = img.shape[1] / img.shape[0]

        if abs(ratio_ref - ratio_img) > ASPECT_TOL:
            return jsonify({"ok": False, "reason": "ASPECT_RATIO_MISMATCH"})

        # resize to template
        img_resized = cv2.resize(img,

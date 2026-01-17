from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# config (quality gates)
MIN_BRIGHTNESS = 60
MAX_BRIGHTNESS = 220
MIN_BLUR_SCORE = 80
ASPECT_TOL = 0.25   # relaxed (message view has UI bars)


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

        h, w = img.shape

        # focus on content area (text region)
        content = img[
            int(h * 0.20):int(h * 0.75),
            int(w * 0.10):int(w * 0.90)
        ]

        # brightness check
        mean_brightness = content.mean()
        if mean_brightness < MIN_BRIGHTNESS:
            return jsonify({"ok": False, "reason": "IMAGE_TOO_DARK"})
        if mean_brightness > MAX_BRIGHTNESS:
            return jsonify({"ok": False, "reason": "IMAGE_TOO_BRIGHT"})

        # blur check
        blur_score = cv2.Laplacian(content, cv2.CV_64F).var()
        if blur_score < MIN_BLUR_SCORE:
            return jsonify({"ok": False, "reason": "IMAGE_TOO_BLURRY"})

        # aspect ratio sanity (very relaxed)
        ratio = w / h
        if ratio < 0.4 or ratio > 0.9:
            return jsonify({"ok": False, "reason": "INVALID_SCREENSHOT_RATIO"})

        # edge sanity (anti blank / edited)
        edges = cv2.Canny(content, 80, 200)
        if edges.mean() < 2:
            return jsonify({"ok": False, "reason": "NO_TEXT_STRUCTURE"})

        # PASSED
        return jsonify({
            "ok": True,
            "quality": {
                "brightness": round(mean_brightness, 1),
                "blur": round(blur_score, 1)
            }
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "reason": "SYSTEM_ERROR",
            "msg": str(e)
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# screenshot profile config
MIN_WIDTH = 360
MIN_HEIGHT = 700
ASPECT_MIN = 0.45   # portrait phone
ASPECT_MAX = 0.80
MIN_EDGE_MEAN = 3.0


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

        # resolution check (anti crop / edited)
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            return jsonify({"ok": False, "reason": "LOW_RESOLUTION"})

        # aspect ratio check (screenshot portrait)
        ratio = w / h
        if ratio < ASPECT_MIN or ratio > ASPECT_MAX:
            return jsonify({"ok": False, "reason": "INVALID_ASPECT_RATIO"})

        # focus on center content
        content = img[
            int(h * 0.15):int(h * 0.85),
            int(w * 0.10):int(w * 0.90)
        ]

        # edge / text structure check
        edges = cv2.Canny(content, 80, 200)
        edge_mean = edges.mean()

        if edge_mean < MIN_EDGE_MEAN:
            return jsonify({"ok": False, "reason": "NO_TEXT_STRUCTURE"})

        # passed screenshot validation
        return jsonify({
            "ok": True,
            "profile": "SCREENSHOT",
            "metrics": {
                "resolution": f"{w}x{h}",
                "aspect_ratio": round(ratio, 2),
                "edge_mean": round(edge_mean, 2)
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

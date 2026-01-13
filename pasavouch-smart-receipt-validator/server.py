from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
from skimage.metrics import structural_similarity as ssim
import os
import uuid

app = Flask(__name__)
CORS(app)

# config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(BASE_DIR, "template_smart_receipt_v1.jpg")
UPLOAD_DIR = os.path.join(BASE_DIR, "temp_upload")
THRESHOLD = 0.90

os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/validate-format", methods=["POST"])
def validate_format():

    if "image" not in request.files:
        return jsonify({ "ok": False, "reason": "NO_IMAGE" })

    file = request.files["image"]

    temp_path = os.path.join(
        UPLOAD_DIR,
        f"{uuid.uuid4().hex}.jpg"
    )

    file.save(temp_path)

    ref = cv2.imread(TEMPLATE, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)

    if ref is None or img is None:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({ "ok": False, "reason": "IMAGE_READ_ERROR" })

    img = cv2.resize(img, (ref.shape[1], ref.shape[0]))
    score, _ = ssim(ref, img, full=True)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    if score >= THRESHOLD:
        return jsonify({
            "ok": True,
            "similarity": round(score, 2)
        })
    else:
        return jsonify({
            "ok": False,
            "reason": "FORMAT_MISMATCH",
            "similarity": round(score, 2)
        })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
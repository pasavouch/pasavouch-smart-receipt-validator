from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import tempfile
import json

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

app = Flask(__name__)
CORS(app)

# Base directory of the server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Reference template image used for format comparison
TEMPLATE_PATH = os.path.join(BASE_DIR, "template_smart_receipt_v1.jpg")

# Load template image once into memory
REF_IMG = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)

# Google Drive folder where receipts will be uploaded
# (replace this with your actual folder ID)
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID")

if not DRIVE_FOLDER_ID:
    raise RuntimeError("DRIVE_FOLDER_ID environment variable not set")


# =========================
# GOOGLE DRIVE AUTH (ENV)
# =========================
# Service account JSON is stored in Render environment variable
service_account_info = json.loads(
    os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
)

credentials = service_account.Credentials.from_service_account_info(
    service_account_info,
    scopes=["https://www.googleapis.com/auth/drive"]
)

drive_service = build("drive", "v3", credentials=credentials)


# =========================
# FORMAT VALIDATION CONFIG
# =========================
THRESHOLD = 0.80
DIFF_LIMIT = 30
EDGE_LIMIT = 15
ASPECT_TOL = 0.12


# =========================
# Function for Image Format Checker
# =========================
# Validates if the uploaded image follows the Smart receipt format
@app.route("/validate-format", methods=["POST"])
def validate_format():

    if "image" not in request.files:
        return jsonify({"ok": False, "reason": "NO_IMAGE"})

    try:
        file = request.files["image"]
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None or REF_IMG is None:
            return jsonify({"ok": False, "reason": "IMAGE_READ_ERROR"})

        h_ref, w_ref = REF_IMG.shape
        ratio_ref = w_ref / h_ref
        ratio_img = img.shape[1] / img.shape[0]

        if abs(ratio_ref - ratio_img) > ASPECT_TOL:
            return jsonify({"ok": False, "reason": "ASPECT_RATIO_MISMATCH"})

        img_resized = cv2.resize(img, (w_ref, h_ref))

        img_proc = cv2.GaussianBlur(img_resized, (5, 5), 0)
        ref_proc = cv2.GaussianBlur(REF_IMG, (5, 5), 0)

        y1, y2 = int(h_ref * 0.27), int(h_ref * 0.77)
        x1, x2 = int(w_ref * 0.22), int(w_ref * 0.78)

        ref_crop = ref_proc[y1:y2, x1:x2]
        img_crop = img_proc[y1:y2, x1:x2]

        wm_y1, wm_y2 = int(h_ref * 0.36), int(h_ref * 0.56)
        wm_x1, wm_x2 = int(w_ref * 0.32), int(w_ref * 0.68)

        edges = cv2.Canny(
            img_resized[wm_y1:wm_y2, wm_x1:wm_x2],
            80,
            200
        )

        if edges.mean() > EDGE_LIMIT:
            return jsonify({"ok": False, "reason": "OVERLAY_DETECTED"})

        if cv2.absdiff(ref_crop, img_crop).mean() > DIFF_LIMIT:
            return jsonify({"ok": False, "reason": "TEMPLATE_DIFF_TOO_HIGH"})

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


# =========================
# Function for Uploading Receipt Image to Google Drive
# =========================
# Called only AFTER JS validation and Firestore save
@app.route("/upload-to-drive", methods=["POST"])
def upload_to_drive():

    if "image" not in request.files:
        return jsonify({"ok": False, "reason": "NO_IMAGE"})

    try:
        image = request.files["image"]
        txn = request.form.get("transactionNumber", "UNKNOWN")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        file_metadata = {
            "name": f"{txn}.jpg",
            "parents": [DRIVE_FOLDER_ID]
        }

        media = MediaFileUpload(
            temp_path,
            mimetype=image.mimetype,
            resumable=False
        )

        drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id"
        ).execute()

        os.remove(temp_path)

        return jsonify({"ok": True})

    except Exception as e:
        return jsonify({
            "ok": False,
            "reason": "UPLOAD_ERROR",
            "msg": str(e)
        })


# Start the Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

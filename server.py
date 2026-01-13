from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
from skimage.metrics import structural_similarity as ssim
import os
import uuid

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(BASE_DIR, "template_smart_receipt_v1.jpg")
UPLOAD_DIR = os.path.join(BASE_DIR, "temp_upload")

THRESHOLD = 0.88

os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/validate-format", methods=["POST"])
def validate_format():

    if "image" not in request.files:
        return jsonify({ "ok": False, "reason": "NO_IMAGE" })

    file = request.files["image"]
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.jpg")
    file.save(temp_path)

    ref = cv2.imread(TEMPLATE, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)

    if ref is None or img is None:
        os.remove(temp_path)
        return jsonify({ "ok": False, "reason": "IMAGE_READ_ERROR" })

    ratio_ref = ref.shape[1] / ref.shape[0]
    ratio_img = img.shape[1] / img.shape[0]

    if abs(ratio_ref - ratio_img) > 0.02:
        os.remove(temp_path)
        return jsonify({ "ok": False, "reason": "ASPECT_RATIO_MISMATCH" })

    img = cv2.resize(img, (ref.shape[1], ref.shape[0]))

    h, w = ref.shape

    y1, y2 = int(h * 0.25), int(h * 0.78)
    x1, x2 = int(w * 0.20), int(w * 0.80)

    ref_crop = ref[y1:y2, x1:x2]
    img_crop = img[y1:y2, x1:x2]

    # overlay / watermark negative zone
    wm_y1, wm_y2 = int(h * 0.35), int(h * 0.55)
    wm_x1, wm_x2 = int(w * 0.30), int(w * 0.70)

    wm_region = img[wm_y1:wm_y2, wm_x1:wm_x2]
    edges = cv2.Canny(wm_region, 80, 200)

    if edges.mean() > 5:
        os.remove(temp_path)
        return jsonify({ "ok": False, "reason": "OVERLAY_DETECTED" })

    diff = cv2.absdiff(ref_crop, img_crop)
    if diff.mean() > 12:
        os.remove(temp_path)
        return jsonify({ "ok": False, "reason": "TEMPLATE_DIFF_TOO_HIGH" })

    ref_edge = cv2.Canny(ref_crop, 80, 200)
    img_edge = cv2.Canny(img_crop, 80, 200)

    score, _ = ssim(ref_edge, img_edge, full=True)
    score = round(score, 2)

    os.remove(temp_path)

    if score >= THRESHOLD:
        return jsonify({ "ok": True, "similarity": score })

    return jsonify({
        "ok": False,
        "reason": "FORMAT_MISMATCH",
        "similarity": score
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

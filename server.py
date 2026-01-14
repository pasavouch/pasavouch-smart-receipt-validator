from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os, json, re, tempfile, random
from datetime import datetime

from google.cloud import vision
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
CORS(app)


def load_env_json(key):
    raw = os.getenv(key)
    if not raw:
        raise RuntimeError(f"Missing ENV variable: {key}")
    return json.loads(raw)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(BASE_DIR, "template_smart_receipt_v1.jpg")
REF_IMG = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)


THRESHOLD = 0.80
DIFF_LIMIT = 30
EDGE_LIMIT = 15
ASPECT_TOL = 0.12


REQUIRED = [
    "you shared your regular load",
    "reference no",
    "load to"
]

FORBIDDEN = [
    "share history",
    "all type",
    "this month",
    "completed",
    "transaction",
    "received"
]


VISION_JSON = load_env_json("GDRIVE_SERVICE_ACCOUNT_JSON")
FIREBASE_JSON = load_env_json("FIREBASE_ADMIN_JSON")
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")
if not GDRIVE_FOLDER_ID:
    raise RuntimeError("Missing ENV variable: GDRIVE_FOLDER_ID")


vision_client = vision.ImageAnnotatorClient.from_service_account_info(VISION_JSON)

drive_creds = service_account.Credentials.from_service_account_info(
    VISION_JSON,
    scopes=["https://www.googleapis.com/auth/drive.file"]
)
drive_service = build("drive", "v3", credentials=drive_creds)


if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_JSON)
    firebase_admin.initialize_app(cred)

db = firestore.client()


def generate_transaction_number():
    date = datetime.now().strftime("%Y%m%d")
    rand = random.randint(100000, 999999)
    return f"PV{date}{rand}"


def compute_converted_cash(amount, plan):
    rate = 0.5
    if plan.upper() == "SILVER":
        rate = 0.7
    elif plan.upper() == "GOLD":
        rate = 0.85
    return round(amount * rate, 2)


def clean_text(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def parse_date(text):
    m = re.search(
        r"(\d{1,2})[\s\-\/](jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\s\-\/](\d{4})",
        text
    )
    if not m:
        return None

    months = {
        "jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "jun": 5,
        "jul": 6, "aug": 7, "sep": 8, "oct": 9, "nov": 10, "dec": 11
    }
    return datetime(int(m.group(3)), months[m.group(2)], int(m.group(1)))


def validate_format(img):
    if REF_IMG is None:
        return False, "TEMPLATE_NOT_LOADED"

    h_ref, w_ref = REF_IMG.shape
    if abs((w_ref / h_ref) - (img.shape[1] / img.shape[0])) > ASPECT_TOL:
        return False, "ASPECT_RATIO_MISMATCH"

    img_resized = cv2.resize(img, (w_ref, h_ref))
    img_proc = cv2.GaussianBlur(img_resized, (5, 5), 0)
    ref_proc = cv2.GaussianBlur(REF_IMG, (5, 5), 0)

    y1, y2 = int(h_ref * 0.27), int(h_ref * 0.77)
    x1, x2 = int(w_ref * 0.22), int(w_ref * 0.78)

    ref_crop = ref_proc[y1:y2, x1:x2]
    img_crop = img_proc[y1:y2, x1:x2]

    wm_y1, wm_y2 = int(h_ref * 0.36), int(h_ref * 0.56)
    wm_x1, wm_x2 = int(w_ref * 0.32), int(w_ref * 0.68)
    edges = cv2.Canny(img_resized[wm_y1:wm_y2, wm_x1:wm_x2], 80, 200)

    if edges.mean() > EDGE_LIMIT:
        return False, "OVERLAY_DETECTED"

    if cv2.absdiff(ref_crop, img_crop).mean() > DIFF_LIMIT:
        return False, "TEMPLATE_DIFF_TOO_HIGH"

    ref_edge = cv2.Canny(ref_crop, 80, 200)
    img_edge = cv2.Canny(img_crop, 80, 200)
    score, _ = ssim(ref_edge, img_edge, full=True)

    if score < THRESHOLD:
        return False, "FORMAT_MISMATCH"

    return True, None


@app.route("/process-receipt", methods=["POST"])
def process_receipt():
    try:
        if "image" not in request.files:
            return jsonify(ok=False, reason="NO_IMAGE")

        file = request.files["image"]
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return jsonify(ok=False, reason="IMAGE_DECODE_FAILED")

        valid, reason = validate_format(img)
        if not valid:
            return jsonify(ok=False, reason=reason)

        vision_image = vision.Image(content=img_bytes.tobytes())
        ocr = vision_client.text_detection(image=vision_image)

        raw_text = ""
        if ocr and ocr.full_text_annotation:
            raw_text = ocr.full_text_annotation.text or ""

        if not raw_text.strip():
            return jsonify(ok=False, reason="NO_OCR_TEXT")

        text = clean_text(raw_text)

        for bad in FORBIDDEN:
            if bad in text:
                return jsonify(ok=False, reason="FORBIDDEN_TEXT")

        for req in REQUIRED:
            if req not in text:
                return jsonify(ok=False, reason="MISSING_REQUIRED_TEXT")

        date = parse_date(text)
        if not date:
            return jsonify(ok=False, reason="DATE_NOT_FOUND")

        if date.date() != datetime.now().date():
            return jsonify(ok=False, reason="INVALID_DATE")

        if not re.search(r"\d{1,2}:\d{2}\s?(am|pm)", text):
            return jsonify(ok=False, reason="TIME_NOT_FOUND")

        amount_match = re.search(r"(₱|p)\s?(\d+(?:\.\d{2})?)", text)
        if not amount_match:
            return jsonify(ok=False, reason="AMOUNT_NOT_FOUND")

        ref_match = re.search(r"\b[a-z0-9]{16}\b", text)
        if not ref_match:
            return jsonify(ok=False, reason="REFERENCE_NOT_FOUND")

        rec_match = re.search(r"(load to|sent to)\s*:?[\s\-]*(09\d{9})", text)
        if not rec_match:
            return jsonify(ok=False, reason="RECIPIENT_NOT_FOUND")

        reference = ref_match.group(0).lower()

        dup = (
            db.collection("pasavouch_smart_reciepts")
            .where("referenceNumber", "==", reference)
            .limit(1)
            .get()
        )

        if dup and len(dup) > 0:
            return jsonify(ok=False, reason="ALREADY_PAID")

        load_amount = float(amount_match.group(2))
        recipient = rec_match.group(2)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(img_bytes.tobytes())
            tmp_path = tmp.name

        media = MediaFileUpload(tmp_path, mimetype="image/jpeg")
        drive_service.files().create(
            body={"name": f"{reference}.jpg", "parents": [GDRIVE_FOLDER_ID]},
            media_body=media
        ).execute()

        os.remove(tmp_path)

        transaction_number = generate_transaction_number()
        converted = compute_converted_cash(load_amount, "Gold")

        db.collection("pasavouch_smart_reciepts").add({
            "transactionNumber": transaction_number,
            "email": "unknown",
            "telcoProvider": "Smart",
            "subscriptionPlan": "Gold",
            "recipientNumber": recipient,
            "referenceNumber": reference,
            "loadAmount": load_amount,
            "convertedToCash": converted,
            "status": "Pending",
            "createdAt": firestore.SERVER_TIMESTAMP
        })

        return jsonify(
            ok=True,
            transactionNumber=transaction_number,
            referenceNumber=reference,
            amount=load_amount
        )

    except Exception as e:
        print("SERVER ERROR:", str(e))
        return jsonify(
            ok=False,
            reason="SERVER_ERROR",
            detail=str(e)
        ), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

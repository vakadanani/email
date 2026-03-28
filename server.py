import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import os
import re
import sqlite3
import json
from functools import wraps
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for  # type: ignore
from flask_cors import CORS  # type: ignore
from werkzeug.security import generate_password_hash, check_password_hash  # type: ignore
import joblib  # type: ignore
import numpy as np  # type: ignore
from typing import Any, Dict, Optional, Tuple, Union
from io import BytesIO
import pytesseract  # type: ignore
from PIL import Image, ImageOps  # type: ignore

_tesseract = os.environ.get("TESSERACT_CMD")
if _tesseract and os.path.isfile(_tesseract):
    pytesseract.pytesseract.tesseract_cmd = _tesseract
elif os.name == "nt":
    _win_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.isfile(_win_tesseract):
        pytesseract.pytesseract.tesseract_cmd = _win_tesseract

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(STATIC_DIR, "spam_history.db")

# ──────────────────────────────────────────────
# OCR (Tesseract)
# ──────────────────────────────────────────────

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

try:
    _LANCZOS = Image.Resampling.LANCZOS  # Pillow 9+
except AttributeError:
    _LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]

# Tesseract works best when glyph size in pixels is moderate; tiny images get upscaled, huge ones downscaled.
OCR_UPSCALE_IF_MAX_SIDE_BELOW = 1000
OCR_DOWNSCALE_IF_MAX_SIDE_ABOVE = 2200
OCR_DOWNSCALE_TARGET_MAX = 2000


def _resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side or m == 0:
        return img
    scale = max_side / m
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((nw, nh), _LANCZOS)


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Grayscale + Otsu threshold to improve Tesseract accuracy on noisy photos."""
    gray = ImageOps.autocontrast(img.convert("L"))
    arr = np.asarray(gray, dtype=np.uint8)
    pixels = arr.ravel()
    hist = np.bincount(pixels, minlength=256).astype(np.float64)
    total = float(pixels.size)
    if total == 0:
        return gray
    sum_all = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0.0
    best_var = 0.0
    thresh = 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > best_var:
            best_var = var_between
            thresh = t
    bw = np.where(arr > thresh, 255, 0).astype(np.uint8)
    return Image.fromarray(bw, mode="L")


def _tesseract_to_string(img: Image.Image, lang: str, tesseract_config: str = "") -> str:
    return pytesseract.image_to_string(img, lang=lang, config=tesseract_config or "").strip()


def ocr_multipass(img: Image.Image, lang: str = "eng") -> str:
    """
    Run several OCR strategies (preprocessing + Tesseract PSM modes, upscale / downscale).
    Returns the longest non-empty result — helps tiny text, large headlines, and huge images.
    """
    best = ""

    def consider(text: str) -> None:
        nonlocal best
        t = (text or "").strip()
        if len(t) > len(best):
            best = t

    w0, h0 = img.size
    m0 = max(w0, h0)

    # Very large images: shrink so letter strokes sit in a range Tesseract handles well
    if m0 > OCR_DOWNSCALE_IF_MAX_SIDE_ABOVE:
        img = _resize_max_side(img, OCR_DOWNSCALE_TARGET_MAX)
    rgb = img.convert("RGB")
    gray = img.convert("L")
    gray_ac = ImageOps.autocontrast(gray)

    # Photos / screenshots: grayscale often beats harsh binarization
    consider(_tesseract_to_string(gray_ac, lang, ""))
    consider(_tesseract_to_string(gray_ac, lang, "--psm 6"))
    consider(_tesseract_to_string(gray_ac, lang, "--psm 3"))
    consider(_tesseract_to_string(gray_ac, lang, "--psm 11"))
    consider(_tesseract_to_string(gray_ac, lang, "--psm 13"))
    # Large headlines, banners, one line of big text, or a single huge word (e.g. "FREE", "WIN")
    consider(_tesseract_to_string(gray_ac, lang, "--psm 7"))
    consider(_tesseract_to_string(gray_ac, lang, "--psm 8"))

    # Sharp black/white text
    otsu = _preprocess_for_ocr(rgb)
    consider(_tesseract_to_string(otsu, lang, ""))
    consider(_tesseract_to_string(otsu, lang, "--psm 6"))
    consider(_tesseract_to_string(otsu, lang, "--psm 11"))
    consider(_tesseract_to_string(otsu, lang, "--psm 7"))
    consider(_tesseract_to_string(otsu, lang, "--psm 8"))

    # Light text on dark backgrounds
    inv = ImageOps.invert(gray_ac)
    consider(_tesseract_to_string(inv, lang, "--psm 6"))
    consider(_tesseract_to_string(inv, lang, "--psm 7"))
    consider(_tesseract_to_string(_preprocess_for_ocr(ImageOps.invert(gray)), lang, "--psm 6"))

    # Small images: upscale then retry (helps tiny or compressed text)
    w, h = img.size
    m = max(w, h)
    if m > 0 and m < OCR_UPSCALE_IF_MAX_SIDE_BELOW:
        scale = min(2.5, OCR_UPSCALE_IF_MAX_SIDE_BELOW / m)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        big = img.resize((nw, nh), _LANCZOS)
        big_ac = ImageOps.autocontrast(big.convert("L"))
        consider(_tesseract_to_string(big_ac, lang, "--psm 6"))
        consider(_tesseract_to_string(big_ac, lang, "--psm 7"))
        consider(_tesseract_to_string(big_ac, lang, "--psm 11"))
        consider(_tesseract_to_string(_preprocess_for_ocr(big.convert("RGB")), lang, "--psm 6"))

    return best


def extract_text_from_image(file: Any, lang: str = "eng") -> str:
    """
    Read an uploaded image file (Werkzeug FileStorage), run multi-pass OCR.
    Raises ValueError for missing/invalid type/unreadable image.
    """
    if file is None or not getattr(file, "filename", None):
        raise ValueError("No image uploaded. Please choose a JPG or PNG file.")
    name = file.filename or ""
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError("Unsupported file type. Please use JPG, JPEG, or PNG.")

    stream = file.stream
    stream.seek(0)
    try:
        img = Image.open(stream)
        img.load()
    except Exception:
        raise ValueError("Could not read this image. It may be corrupted or not a valid image.") from None

    return ocr_multipass(img, lang=lang)


def ocr_image(
    source: Union[str, bytes, bytearray, Image.Image],
    lang: str = "eng",
) -> str:
    """Return plain text from an image path, bytes, or PIL Image (multi-pass OCR)."""
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, (bytes, bytearray)):
        img = Image.open(BytesIO(source))
    elif isinstance(source, str):
        img = Image.open(source)
    else:
        raise TypeError("source must be a file path str, bytes, bytearray, or PIL.Image")

    return ocr_multipass(img, lang=lang)

app = Flask(__name__)
app.secret_key = 'spamshield-ai-secret-key-2026'
CORS(app)

# ──────────────────────────────────────────────
# SQLite Initialisation
# ──────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            message_preview TEXT,
            full_message    TEXT,
            prediction      TEXT,
            confidence      REAL,
            spam_words      TEXT,
            urls_found      TEXT,
            explanation     TEXT,
            sender          TEXT DEFAULT '',
            feedback        TEXT DEFAULT '',
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email    TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS email_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            email_text  TEXT,
            prediction  TEXT,
            confidence  REAL,
            spam_words  TEXT,
            urls_found  TEXT,
            explanation TEXT,
            sender      TEXT DEFAULT '',
            feedback    TEXT DEFAULT '',
            timestamp   TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


init_db()

# ──────────────────────────────────────────────
# Load ML Artefacts
# ──────────────────────────────────────────────
model: Any = None
vectorizer: Any = None
model_accuracy: Any = None
feature_names: Any = None

try:
    model = joblib.load(os.path.join(STATIC_DIR, "model.pkl"))
    vectorizer = joblib.load(os.path.join(STATIC_DIR, "vectorizer.pkl"))
    model_accuracy = joblib.load(os.path.join(STATIC_DIR, "accuracy.pkl"))
    fn_path = os.path.join(STATIC_DIR, "feature_names.pkl")
    if os.path.exists(fn_path):
        feature_names = joblib.load(fn_path)
    print(f"Model loaded. Accuracy: {model_accuracy:.4f}")
except Exception as e:
    print(f"Error loading model: {e}")

# ──────────────────────────────────────────────
# BERT / Transformers (loaded once at startup)
# ──────────────────────────────────────────────
bert_classifier: Any = None
BERT_MODEL_ID = os.environ.get(
    "SPAM_BERT_MODEL",
    "mrm8488/bert-tiny-finetuned-sms-spam-detection",
)

try:
    import torch  # type: ignore
    from transformers import pipeline  # type: ignore

    _bert_device = 0 if torch.cuda.is_available() else -1
    bert_classifier = pipeline(
        "text-classification",
        model=BERT_MODEL_ID,
        tokenizer=BERT_MODEL_ID,
        device=_bert_device,
    )
    print(f"BERT classifier loaded: {BERT_MODEL_ID} (device={_bert_device})")
except Exception as e:
    bert_classifier = None
    print(f"BERT not loaded — using TF-IDF + Naive Bayes only: {e}")

# ──────────────────────────────────────────────
# Auth Helpers
# ──────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            # For API endpoints return 401; for page requests redirect
            if request.path.startswith('/api/') or request.is_json or request.headers.get('Accept', '').startswith('application/json'):
                return jsonify({'error': 'Unauthorized'}), 401
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated


# ──────────────────────────────────────────────
# Suspicious URL / Phishing helpers
# ──────────────────────────────────────────────
URL_REGEX = re.compile(
    r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', re.IGNORECASE
)

SUSPICIOUS_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "cutt.ly",
    "free-money.com", "prize-winner.com", "claim-now.net",
}

SUSPICIOUS_TLDS = {".xyz", ".top", ".buzz", ".club", ".work", ".click", ".loan", ".win"}


def analyse_urls(text):
    urls = URL_REGEX.findall(text)
    results = []
    for url in urls:
        domain = url.split("//")[-1].split("/")[0].split("?")[0].lower()
        flags = []
        if domain in SUSPICIOUS_DOMAINS:
            flags.append("Known URL shortener / suspicious domain")
        if any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS):
            flags.append("Suspicious TLD")
        if re.search(r'\d{1,3}(\.\d{1,3}){3}', domain):
            flags.append("IP-based URL")
        if len(domain) > 40:
            flags.append("Unusually long domain")
        results.append({"url": url, "domain": domain, "flags": flags, "suspicious": len(flags) > 0})
    return results

# ──────────────────────────────────────────────
# Rule-based spam keywords (Hybrid Detection)
# ──────────────────────────────────────────────
SPAM_RULES = [
    (r'\bfree\s+money\b', "Contains 'free money'"),
    (r'\byou\s+have\s+won\b', "Contains 'you have won'"),
    (r'\bclaim\s+(your|now|prize)\b', "Contains 'claim your/now/prize'"),
    (r'\burgent\b', "Uses urgency language"),
    (r'\bact\s+now\b', "Contains 'act now'"),
    (r'\blimited\s+time\b', "Contains 'limited time'"),
    (r'\bclick\s+here\b', "Contains 'click here'"),
    (r'\bregistration\s+fee\b', "Contains 'registration fee'"),
    (r'\bbank\s+details\b', "Asks for bank details"),
    (r'\b(earn|make)\s+\$?\d', "Promises specific earnings"),
    (r'\bcongratulations\b', "Contains 'congratulations'"),
    (r'\b(bitcoin|btc|crypto)\b', "References cryptocurrency"),
    (r'\bno\s+investment\b', "Claims no investment needed"),
    (r'\b100%\s+free\b', "Claims 100% free"),
    (r'\bunsubscribe\b', "Contains 'unsubscribe'"),
]


def rule_based_check(text):
    lower = text.lower()
    triggered = []
    for pattern, desc in SPAM_RULES:
        if re.search(pattern, lower):
            triggered.append(desc)
    return triggered

# ──────────────────────────────────────────────
# Explainable AI — top contributing features
# ──────────────────────────────────────────────

def get_spam_words(text, top_n=10):
    """Return the top N words in the text that most strongly indicate spam."""
    if feature_names is None or model is None:
        return []

    vec = vectorizer.transform([text])
    # log-probability per feature for the 'spam' class
    classes = list(model.classes_)
    spam_idx = classes.index("spam") if "spam" in classes else 1
    log_probs = model.feature_log_prob_[spam_idx]

    # Features present in this message
    nonzero = vec.nonzero()[1]
    word_scores: list[tuple[str, float]] = []
    for idx in nonzero:
        word = feature_names[idx]
        # Only keep words that actually appear in the original text (case-insensitive)
        if word.lower() in text.lower():
            word_scores.append((word, float(log_probs[idx])))

    word_scores.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in word_scores[:top_n]]  # pyre-ignore[16]


def build_explanation(prediction, confidence, spam_words, rule_hits, url_results):
    reasons = []
    if prediction == "spam":
        if confidence > 0.85:
            reasons.append(f"High spam probability ({confidence*100:.0f}%)")
        if spam_words:
            reasons.append(f"Contains spam-associated words: {', '.join(spam_words[:5])}")
        if rule_hits:
            for r in rule_hits[:3]:
                reasons.append(r)
        sus_urls = [u for u in url_results if u["suspicious"]]
        if sus_urls:
            reasons.append(f"Contains {len(sus_urls)} suspicious link(s)")
    else:
        reasons.append("No significant spam patterns detected")
        if confidence > 0.9:
            reasons.append(f"High confidence safe email ({confidence*100:.0f}%)")
    return reasons


# ──────────────────────────────────────────────
# Extract sender from email text
# ──────────────────────────────────────────────

def extract_sender(text):
    match = re.search(r'(?:from|sender)[:\s]+([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    # fallback: any email address
    match = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    return match.group(0).lower() if match else ""


def _resolve_pipeline_label(label_val: Any, id2l: Dict[Any, Any]) -> str:
    """Map LABEL_0 / ids to human label from model config."""
    s = str(label_val).upper()
    if s.startswith("LABEL_"):
        try:
            idx = int(s.rsplit("_", 1)[-1])
            return str(id2l.get(idx, id2l.get(str(idx), label_val)))
        except (ValueError, IndexError):
            pass
    return str(label_val)


def _label_means_spam(name: str) -> bool:
    u = name.upper()
    if "HAM" in u or "NOT SPAM" in u or "LEGITIMATE" in u or u in ("N", "NO", "NEG", "NEGATIVE"):
        return False
    if "SPAM" in u:
        return True
    return False


def _flatten_classification_output(batch: Any) -> list:
    """Normalize pipeline output to a list of {label, score} dicts."""
    if not batch:
        return []
    if isinstance(batch, dict):
        return [batch]
    if not isinstance(batch, list):
        return []
    first = batch[0]
    if isinstance(first, dict):
        return batch
    if isinstance(first, list):
        return first
    return []


def classify_with_bert(text: str) -> Dict[str, Any]:
    """
    Run Hugging Face text-classification pipeline on raw text (truncated to model max length).
    Returns display label, internal spam/ham, and confidence in [0, 1].
    """
    if bert_classifier is None:
        raise RuntimeError("BERT classifier is not loaded")

    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty text")

    cfg = bert_classifier.model.config
    id2l = getattr(cfg, "id2label", None) or {}
    n_cls = len(id2l) if id2l else 2

    batch = bert_classifier(raw, truncation=True, max_length=512, top_k=min(n_cls, 8))
    rows = _flatten_classification_output(batch)

    best_spam = 0.0
    best_ham = 0.0
    for row in rows:
        resolved = _resolve_pipeline_label(row.get("label"), id2l)
        sc = float(row.get("score", 0.0))
        if _label_means_spam(resolved):
            best_spam = max(best_spam, sc)
        else:
            best_ham = max(best_ham, sc)

    if best_spam == 0.0 and best_ham == 0.0 and rows:
        resolved = _resolve_pipeline_label(rows[0].get("label"), id2l)
        sc = float(rows[0].get("score", 0.0))
        pred_raw = "spam" if _label_means_spam(resolved) else "ham"
        return {
            "prediction": "Spam" if pred_raw == "spam" else "Not Spam",
            "prediction_raw": pred_raw,
            "confidence": min(1.0, max(0.0, sc)),
        }

    if best_spam >= best_ham:
        conf = best_spam if best_spam > 0 else (1.0 - best_ham)
        return {"prediction": "Spam", "prediction_raw": "spam", "confidence": min(1.0, max(0.0, conf))}

    conf = best_ham if best_ham > 0 else (1.0 - best_spam)
    return {"prediction": "Not Spam", "prediction_raw": "ham", "confidence": min(1.0, max(0.0, conf))}


def legacy_ml_predict(message: str) -> Tuple[str, float]:
    """TF-IDF + Naive Bayes only (no rules)."""
    if model is None or vectorizer is None or not hasattr(model, "predict"):
        raise RuntimeError("Legacy model not loaded")
    transformed = vectorizer.transform([message])
    prediction = model.predict(transformed)[0]
    proba = model.predict_proba(transformed)[0]
    return prediction, float(max(proba))


def analyze_message_for_spam(message: str, sender: str = "") -> Dict[str, Any]:
    """TF-IDF + Naive Bayes, hybrid rules, URLs, explanations. Does not write to DB."""
    prediction, confidence = legacy_ml_predict(message)
    rule_hits = rule_based_check(message)
    url_results = analyse_urls(message)
    spam_words = get_spam_words(message)

    if prediction == "ham" and len(rule_hits) >= 3:
        prediction = "spam"
        confidence = max(confidence, 0.72)

    if not sender:
        sender = extract_sender(message)

    explanation = build_explanation(prediction, confidence, spam_words, rule_hits, url_results)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "spam_words": spam_words,
        "urls": url_results,
        "rule_hits": rule_hits,
        "explanation": explanation,
        "sender": sender,
        "classifier": "TF-IDF + NB",
    }


def analyze_message(
    message: str,
    sender: str = "",
    prefer_bert: bool = True,
) -> Dict[str, Any]:
    """
    Prefer BERT when loaded; on failure fall back to TF-IDF + NB.
    Always applies rule/URL hybrid boost and explainability.
    """
    rule_hits = rule_based_check(message)
    url_results = analyse_urls(message)
    if not sender:
        sender = extract_sender(message)

    classifier_name = "TF-IDF + NB"
    bert_fail: Optional[str] = None
    prediction = "ham"
    confidence = 0.5

    if prefer_bert and bert_classifier is not None:
        try:
            bc = classify_with_bert(message)
            prediction = bc["prediction_raw"]
            confidence = bc["confidence"]
            classifier_name = "BERT"
        except Exception as ex:
            bert_fail = str(ex)

    if classifier_name != "BERT":
        try:
            prediction, confidence = legacy_ml_predict(message)
        except RuntimeError as err:
            if bert_fail:
                raise RuntimeError(f"BERT failed ({bert_fail}) and legacy model: {err}") from err
            raise

    spam_words = get_spam_words(message) if model is not None and feature_names is not None else []

    if prediction == "ham" and len(rule_hits) >= 3:
        prediction = "spam"
        confidence = max(confidence, 0.72)

    explanation = build_explanation(prediction, confidence, spam_words, rule_hits, url_results)
    if classifier_name == "BERT":
        explanation.insert(0, "BERT transformer classification (contextual semantics)")
    elif bert_fail:
        explanation.insert(0, f"BERT unavailable ({bert_fail}); used TF-IDF + Naive Bayes")

    return {
        "prediction": prediction,
        "confidence": confidence,
        "spam_words": spam_words,
        "urls": url_results,
        "rule_hits": rule_hits,
        "explanation": explanation,
        "sender": sender,
        "classifier": classifier_name,
    }


def persist_scan_result(message: str, analysis: Dict[str, Any]) -> int:
    """Insert global scan_history and per-user email_history; return scan_history id."""
    preview = message[:120] + ("..." if len(message) > 120 else "")
    spam_words = analysis["spam_words"]
    url_results = analysis["urls"]
    explanation = analysis["explanation"]
    prediction = analysis["prediction"]
    confidence = analysis["confidence"]
    sender = analysis["sender"]

    conn = get_db()
    cursor = conn.execute(
        """INSERT INTO scan_history
           (message_preview, full_message, prediction, confidence, spam_words, urls_found, explanation, sender)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            preview,
            message,
            prediction,
            round(confidence, 4),
            json.dumps(spam_words),
            json.dumps(url_results),
            json.dumps(explanation),
            sender,
        ),
    )
    scan_id = cursor.lastrowid
    user_id = session["user_id"]
    conn.execute(
        """INSERT INTO email_history
           (user_id, email_text, prediction, confidence, spam_words, urls_found, explanation, sender)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            message,
            prediction,
            round(confidence, 4),
            json.dumps(spam_words),
            json.dumps(url_results),
            json.dumps(explanation),
            sender,
        ),
    )
    conn.commit()
    conn.close()
    return scan_id


# ══════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════

@app.route('/login')
def login_page():
    if 'user_id' in session:
        return redirect('/')
    return send_from_directory(STATIC_DIR, 'login.html')


@app.route('/register')
def register_page():
    if 'user_id' in session:
        return redirect('/')
    return send_from_directory(STATIC_DIR, 'register.html')


@app.route('/api/register', methods=['POST'])
def api_register():
    try:
        data = request.json
        username = (data.get('username') or '').strip()
        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''
        confirm = data.get('confirm_password') or ''

        if not username or not email or not password:
            return jsonify({'error': 'All fields are required'}), 400
        if len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters'}), 400
        if '@' not in email or '.' not in email:
            return jsonify({'error': 'Invalid email address'}), 400
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        if password != confirm:
            return jsonify({'error': 'Passwords do not match'}), 400

        conn = get_db()
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            conn.close()
            return jsonify({'error': 'An account with this email already exists'}), 409

        hashed = generate_password_hash(password)
        cursor = conn.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, hashed)
        )
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()

        session['user_id'] = user_id
        session['username'] = username
        session['email'] = email

        return jsonify({'success': True, 'username': username})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        data = request.json
        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''

        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

        if not user or not check_password_hash(user['password'], password):
            return jsonify({'error': 'Invalid email or password'}), 401

        session['user_id'] = user['id']
        session['username'] = user['username']
        session['email'] = user['email']

        return jsonify({'success': True, 'username': user['username']})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/logout')
def api_logout():
    session.clear()
    return redirect('/login')


@app.route('/api/me')
def api_me():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    return jsonify({
        'user_id': session['user_id'],
        'username': session['username'],
        'email': session['email']
    })


# ══════════════════════════════════════════════
# MAIN APP ROUTES (Protected)
# ══════════════════════════════════════════════

@app.route('/')
@login_required
def index():
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/<path:filename>')
def static_files(filename):
    # Allow auth pages and static assets without login
    public_files = {'login.html', 'register.html', 'style.css', 'app.js',
                    'safe_mail_tone.mp3', 'warning_beep.mp3'}
    if filename in public_files or filename.endswith(('.css', '.js', '.mp3', '.png', '.ico', '.svg', '.woff', '.woff2', '.ttf')):
        return send_from_directory(STATIC_DIR, filename)
    # Everything else requires login
    if 'user_id' not in session:
        return redirect('/login')
    return send_from_directory(STATIC_DIR, filename)

# ─── Predict ──────────────────────────────────

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        message = ""
        use_bert = True
        sender_override = ""
        ct = (request.content_type or "")

        if "multipart/form-data" in ct:
            message = (request.form.get("message") or "").strip()
            sender_override = (request.form.get("sender") or "").strip()
            ub = (request.form.get("use_bert") or "true").lower()
            use_bert = ub not in ("0", "false", "no", "off")
            if "image" in request.files and request.files["image"].filename:
                img_f = request.files["image"]
                try:
                    ocr_t = extract_text_from_image(img_f)
                except ValueError as err:
                    return jsonify({"error": str(err)}), 400
                except pytesseract.TesseractNotFoundError:
                    return jsonify({
                        "error": "OCR (Tesseract) is not available. Install Tesseract or set TESSERACT_CMD.",
                    }), 503
                if ocr_t and message:
                    message = f"{message} {ocr_t}".strip()
                elif ocr_t:
                    message = ocr_t.strip()
                elif not message:
                    return jsonify({
                        "error": "No text to analyze. Add email text and/or an image with readable text.",
                    }), 400
        else:
            data = request.json
            if not data or "message" not in data:
                return jsonify({"error": "No message provided"}), 400
            message = (data.get("message") or "").strip()
            sender_override = (data.get("sender") or "").strip()
            use_bert = data.get("use_bert", True)
            if isinstance(use_bert, str):
                use_bert = use_bert.lower() not in ("false", "0", "no", "off")

        if not message.strip():
            return jsonify({"error": "No text to analyze. Paste an email or attach an image with text."}), 400

        sender = sender_override or extract_sender(message)

        if len(message) > 5000:
            return jsonify({"error": "Message too long (max 5000 characters)"}), 400

        try:
            analysis = analyze_message(message, sender, prefer_bert=use_bert)
        except RuntimeError as err:
            return jsonify({"error": str(err)}), 500

        scan_id = persist_scan_result(message, analysis)

        pred_label = "Spam" if analysis["prediction"] == "spam" else "Not Spam"
        return jsonify({
            "id": scan_id,
            "prediction": analysis["prediction"],
            "prediction_label": pred_label,
            "confidence": round(analysis["confidence"], 3),
            "confidence_percent": round(analysis["confidence"] * 100, 1),
            "model": analysis["classifier"],
            "classifier": analysis["classifier"],
            "model_accuracy": round(model_accuracy, 4) if model_accuracy else None,
            "spam_words": analysis["spam_words"],
            "urls": analysis["urls"],
            "rule_hits": analysis["rule_hits"],
            "explanation": analysis["explanation"],
            "sender": analysis["sender"],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict_bert', methods=['POST'])
@login_required
def predict_bert():
    """Same as JSON /predict with BERT preferred; accepts only application/json."""
    try:
        data = request.json
        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"error": "Empty message"}), 400
        sender = (data.get("sender") or "").strip() or extract_sender(message)
        if len(message) > 5000:
            return jsonify({"error": "Message too long (max 5000 characters)"}), 400

        try:
            analysis = analyze_message(message, sender, prefer_bert=True)
        except RuntimeError as err:
            return jsonify({"error": str(err)}), 500

        scan_id = persist_scan_result(message, analysis)
        pred_label = "Spam" if analysis["prediction"] == "spam" else "Not Spam"
        return jsonify({
            "prediction": pred_label,
            "confidence": round(analysis["confidence"] * 100, 1),
            "model": analysis["classifier"],
            "id": scan_id,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict_image', methods=['POST'])
@login_required
def predict_image():
    """OCR an uploaded image, optionally merge with email text, run spam model."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded. Please choose a JPG or PNG file.'}), 400

        img_file = request.files['image']
        optional_text = (request.form.get('email_text') or request.form.get('message') or '').strip()

        try:
            ocr_text = extract_text_from_image(img_file)
        except ValueError as err:
            return jsonify({'error': str(err)}), 400
        except pytesseract.TesseractNotFoundError:
            return jsonify({
                'error': 'OCR engine (Tesseract) is not installed or not configured. Set TESSERACT_CMD if needed.',
            }), 503

        if not ocr_text:
            return jsonify({
                'error': (
                    'No text could be read from this image after trying multiple OCR modes '
                    '(including headline/single-line modes and resizing for very large or small images). '
                    'Try a clearer, well-lit photo, less glare, or crop tightly around the words.'
                ),
            }), 400

        if optional_text:
            combined = f"{optional_text} {ocr_text}".strip()
        else:
            combined = ocr_text

        if len(combined) > 5000:
            return jsonify({'error': 'Combined email text and image text exceed 5000 characters.'}), 400

        sender = extract_sender(combined)
        try:
            analysis = analyze_message(combined, sender, prefer_bert=True)
        except RuntimeError as err:
            return jsonify({'error': str(err)}), 500

        scan_id = persist_scan_result(combined, analysis)

        pred_label = "Spam" if analysis['prediction'] == "spam" else "Not Spam"
        conf_pct = round(analysis['confidence'] * 100, 1)

        return jsonify({
            'extracted_text': ocr_text,
            'prediction': pred_label,
            'confidence': conf_pct,
            'id': scan_id,
            'model': analysis['classifier'],
            'classifier': analysis['classifier'],
            'model_accuracy': round(model_accuracy, 4) if model_accuracy else None,
            'combined_with_email': bool(optional_text),
            'prediction_raw': analysis['prediction'],
            'explanation': analysis['explanation'],
            'spam_words': analysis['spam_words'],
            'urls': analysis['urls'],
            'sender': analysis['sender'],
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── File Upload ──────────────────────────────

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        allowed_ext = {'.txt', '.eml', '.msg'}
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_ext:
            return jsonify({'error': f'Unsupported file type. Use: {", ".join(allowed_ext)}'}), 400

        text = file.read().decode('utf-8', errors='ignore')
        if len(text.strip()) == 0:
            return jsonify({'error': 'File is empty'}), 400

        return jsonify({'text': text[:5000], 'filename': file.filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Feedback ─────────────────────────────────

@app.route('/feedback', methods=['POST'])
@login_required
def feedback():
    try:
        data = request.json
        scan_id = data.get('id')
        fb = data.get('feedback')  # "spam" or "ham"
        if not scan_id or fb not in ("spam", "ham"):
            return jsonify({'error': 'Invalid feedback'}), 400

        conn = get_db()
        conn.execute("UPDATE scan_history SET feedback = ? WHERE id = ?", (fb, scan_id))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── History / Dashboard (per-user) ──────────

@app.route('/history', methods=['GET'])
@login_required
def history():
    try:
        user_id = session['user_id']
        conn = get_db()
        rows = conn.execute(
            """SELECT id, email_text, prediction, confidence, sender, feedback, timestamp
               FROM email_history WHERE user_id = ? ORDER BY id DESC LIMIT 50""",
            (user_id,)
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM email_history WHERE user_id = ?", (user_id,)).fetchone()[0]
        spam_count = conn.execute("SELECT COUNT(*) FROM email_history WHERE user_id = ? AND prediction='spam'", (user_id,)).fetchone()[0]
        ham_count = conn.execute("SELECT COUNT(*) FROM email_history WHERE user_id = ? AND prediction='ham'", (user_id,)).fetchone()[0]
        conn.close()

        history_data = []
        for r in rows:
            row_dict = dict(r)
            # Create message_preview from email_text
            email_text = row_dict.get('email_text', '') or ''
            row_dict['message_preview'] = email_text[:120] + ('...' if len(email_text) > 120 else '')
            row_dict['created_at'] = row_dict.pop('timestamp', '')
            history_data.append(row_dict)

        return jsonify({
            'history': history_data,
            'stats': {'total': total, 'spam': spam_count, 'ham': ham_count}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Analytics (per-user) ────────────────────

@app.route('/analytics', methods=['GET'])
@login_required
def analytics():
    try:
        user_id = session['user_id']
        conn = get_db()
        daily = conn.execute("""
            SELECT date(timestamp) as day,
                   COUNT(*) as total,
                   SUM(CASE WHEN prediction='spam' THEN 1 ELSE 0 END) as spam,
                   SUM(CASE WHEN prediction='ham' THEN 1 ELSE 0 END) as ham
            FROM email_history
            WHERE user_id = ? AND timestamp >= datetime('now', '-30 days')
            GROUP BY date(timestamp)
            ORDER BY day
        """, (user_id,)).fetchall()
        conn.close()

        return jsonify({
            'daily': [dict(r) for r in daily]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Sender Reputation ───────────────────────

@app.route('/reputation', methods=['GET'])
@login_required
def reputation():
    try:
        sender = request.args.get('sender', '').strip().lower()
        if not sender:
            return jsonify({'error': 'No sender provided'}), 400

        user_id = session['user_id']
        conn = get_db()
        row = conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN prediction='spam' THEN 1 ELSE 0 END) as spam,
                   SUM(CASE WHEN prediction='ham' THEN 1 ELSE 0 END) as ham
            FROM email_history WHERE user_id = ? AND sender = ?
        """, (user_id, sender)).fetchone()
        conn.close()

        total = row['total'] or 0
        spam = row['spam'] or 0
        spam_ratio = (spam / total * 100) if total > 0 else 0

        if total == 0:
            level = "unknown"
        elif spam_ratio > 60:
            level = "dangerous"
        elif spam_ratio > 30:
            level = "suspicious"
        else:
            level = "trusted"

        return jsonify({
            'sender': sender,
            'total_scans': total,
            'spam_count': spam,
            'ham_count': row['ham'] or 0,
            'spam_ratio': round(spam_ratio, 1),
            'level': level
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting SpamShield AI on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
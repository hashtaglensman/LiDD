# app.py
"""Flask entry‑point for DeepFake Detector (Heroku‑/Docker‑ready).

Expects a POSTed .mp4 file (key: "video") from the HTML form or API.
Runs the optimised inference pipeline and returns a human‑readable label.

To run locally:
    $ gunicorn app:app  # Heroku uses this by default if Procfile present
"""
import os
import tempfile
from flask import Flask, request, render_template, jsonify
from opt_inf_pip import pred_cred  # ← import from your pipeline module

# ---------------------------------------------------------------------------
# Flask setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

ALLOWED_EXT = {"mp4"}

def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    """Landing page with the upload form (templates/home.html)."""
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle form submission from the browser."""
    if "video" not in request.files:
        return render_template("home.html", pred="❌ No file part in request.")

    f = request.files["video"]
    if f.filename == "":
        return render_template("home.html", pred="❌ No file selected.")

    if not _allowed(f.filename):
        return render_template(
            "home.html", pred="❌ Only .mp4 files are supported.")

    # Save to a temp file (Heroku’s filesystem is ephemeral but fine for inference)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        f.save(tmp.name)
        video_path = tmp.name

    try:
        raw_pred = pred_cred(video_path)
        label = "Real" if int(raw_pred) == 0 else "Fake"
        message = f"The content may be {label}."
    except Exception as e:
        message = f"❌ Inference error: {e}"
    finally:
        try:
            os.remove(video_path)
        except FileNotFoundError:
            pass

    return render_template("home.html", pred=message)


@app.route("/predict_api", methods=["POST"])
def predict_api():
    """Machine‑friendly endpoint.

    Accepts multipart/form‑data (key "video") *or* JSON with {"video_path": "..."}.
    Returns JSON {"label": "Real|Fake", "raw": 0|1}.
    """
    # 1️⃣ Multipart upload path (e.g. Postman, cURL ‑F)
    if "video" in request.files:
        f = request.files["video"]
        if f.filename == "" or not _allowed(f.filename):
            return jsonify({"error": "Invalid or missing .mp4 file."}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            f.save(tmp.name)
            video_path = tmp.name
        cleanup = True

    # 2️⃣ JSON path ‑ user supplies a server‑accessible file path
    else:
        payload = request.get_json(silent=True) or {}
        video_path = payload.get("video_path")
        if not video_path or not os.path.exists(video_path):
            return jsonify({"error": "Provide a valid 'video_path' or multipart 'video'."}), 400
        cleanup = False  # user‑provided path

    try:
        raw_pred = pred_cred(video_path)
        label = "Real" if int(raw_pred) == 0 else "Fake"
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if cleanup:
            try:
                os.remove(video_path)
            except FileNotFoundError:
                pass

    return jsonify({"label": label, "raw": int(raw_pred)})


# ---------------------------------------------------------------------------
# Entry‑point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Heroku sets PORT env var automatically; default to 5000 locally
    port = int(os.environ.get("PORT", 5005))
    app.run(host="0.0.0.0", port=port)

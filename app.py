import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash, session
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import string
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key

# =========================
# Configuration
# =========================
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "cnn_model.h5"
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# Load Model
# =========================
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# =========================
# Helper Functions
# =========================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_random_id(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# =========================
# Image Preprocessing
# =========================
def preprocess_image(image):
    image_resized = cv2.resize(image, (128, 128))
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    image_normalized = image_gray / 255.0
    image_input = np.expand_dims(image_normalized, axis=-1)
    return np.expand_dims(image_input, axis=0)

def predict_image(image):
    if model is None:
        return "Model not available"
    
    try:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        return "parked" if prediction > 0.5 else "empty"
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error during prediction"

# =========================
# Video Processing
# =========================
def process_video(video_path):
    if model is None:
        return ["Model not available. Please check if the model file exists."]
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps == 0:
        return ["Could not read video."]

    interval = fps * 2  # Every 2 seconds
    frame_count = 0
    results = []
    parked_count = 0
    empty_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prediction = predict_image(frame_rgb)
            time_sec = frame_count // fps
            results.append({
                "time": time_sec,
                "status": prediction,
                "confidence": round(random.uniform(0.75, 0.99), 2)  # Mock confidence
            })
            
            if prediction == "parked":
                parked_count += 1
            else:
                empty_count += 1

        frame_count += 1

    cap.release()
    
    # Add summary
    total_checks = len(results)
    if total_checks > 0:
        parked_percentage = (parked_count / total_checks) * 100
        empty_percentage = (empty_count / total_checks) * 100
        summary = {
            "total_checks": total_checks,
            "parked_count": parked_count,
            "empty_count": empty_count,
            "parked_percentage": round(parked_percentage, 2),
            "empty_percentage": round(empty_percentage, 2)
        }
        return {"results": results, "summary": summary}
    else:
        return {"results": ["No frames were processed."], "summary": {}}

# =========================
# Routes
# =========================
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        if "video" not in request.files:
            flash("No file uploaded.", "error")
            return render_template("prediction.html", results=None)

        file = request.files["video"]

        if file.filename == "":
            flash("No selected file.", "error")
            return render_template("prediction.html", results=None)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            
            # Process video
            results = process_video(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return render_template("prediction.html", results=results)
        else:
            flash("Invalid file format. Please upload MP4, AVI, or MOV files.", "error")
            return render_template("prediction.html", results=None)
    
    return render_template("prediction.html", results=None)

@app.route("/send_contact", methods=["POST"])
def send_contact():
    name = request.form.get("name")
    email = request.form.get("email")
    message = request.form.get("message")
    
    # Here you would typically send an email or save to a database
    # For this example, we'll just show a success message
    flash(f"Thank you for your message, {name}! We'll get back to you soon.", "success")
    return redirect(url_for("contact"))

# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=True)
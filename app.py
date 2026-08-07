import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, render_template, request
import pickle
from features import extract_features

app = Flask(__name__)

# Load the trained model once, when the server starts (not on every request -- faster)
with open("outputs/emotion_model.pkl", "rb") as f:
    saved = pickle.load(f)
    model = saved["model"]
    scaler = saved["scaler"]

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("audio_file")
    if file is None or file.filename == "":
        return render_template("index.html", prediction="No file uploaded")

    # Save the uploaded file temporarily so librosa/soundfile can read it
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Run it through our existing feature extraction + model
    features = extract_features(file_path)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]

    # Also get confidence scores for each emotion
    probabilities = model.predict_proba(features_scaled)[0]
    emotion_probs = dict(zip(model.classes_, probabilities))

    os.remove(file_path)  # cleanup

    return render_template("index.html", prediction=prediction, probabilities=emotion_probs)


if __name__ == "__main__":
    app.run(debug=True)
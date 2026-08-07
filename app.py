import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, render_template, request
import pickle
from features import extract_features

from dotenv import load_dotenv
from google import genai

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

# Load the trained model once, when the server starts (not on every request -- faster)
with open("outputs/emotion_model.pkl", "rb") as f:
    saved = pickle.load(f)
    model = saved["model"]
    scaler = saved["scaler"]

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_empathetic_response(emotion):
    """
    Sends the detected emotion to Gemini and gets back a short,
    natural-language suggestion for how to respond to someone feeling that way.
    """
    try:
        prompt = (
            f"Someone's voice was just analyzed and detected as sounding '{emotion}'. "
            f"In 1-2 short sentences, suggest a kind, empathetic way someone could "
            f"respond to them. Keep it natural and conversational, not clinical."
        )
        response = gemini_client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"[Gemini error] {e}")  # visible in terminal for debugging, not shown to users
        return "Suggestion temporarily unavailable — please try again in a moment."
        


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

    # Ask Gemini for a suggested empathetic response based on the detected emotion
    suggestion = get_empathetic_response(prediction)

    return render_template(
        "index.html",
        prediction=prediction,
        probabilities=emotion_probs,
        suggestion=suggestion,
    )


if __name__ == "__main__":
    app.run(debug=True)
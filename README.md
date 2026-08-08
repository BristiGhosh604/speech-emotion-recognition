<div align="center">

<!-- Animated typing header -->
<img src="https://readme-typing-svg.demolab.com?font=Poppins&size=32&duration=3000&pause=1000&color=764ABA&center=true&vCenter=true&width=600&lines=Speech+Emotion+Recognition;Classical+ML+%2B+LLM+Integration;Rebuilt+From+Scratch" alt="Typing SVG" />

<br/>

🎙️ **Detect emotion from speech — trained on RAVDESS, powered by Random Forest + Gemini LLM, deployed live.**

<br/>

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Try_it_now-764ABA?style=for-the-badge)](https://speech-emotion-recognition-wadx.onrender.com)
[![GitHub Repo](https://img.shields.io/badge/📂_Source-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/BristiGhosh604/speech-emotion-recognition)

<br/>

<!-- Tech stack badges -->
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Gemini_API-8E75B2?style=flat-square&logo=googlegemini&logoColor=white)
![FFmpeg](https://img.shields.io/badge/FFmpeg-007808?style=flat-square&logo=ffmpeg&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)
![Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7?style=flat-square&logo=render&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white)

</div>

<br/>

## 📖 About

This started as a university final-year group project. After graduating, I rebuilt it entirely from scratch — new architecture, new code, modern engineering practices — to reflect current industry standards rather than a classroom exercise.

**What it does:** upload an audio clip or record live from your browser, and the app detects the speaker's emotion (angry, calm, happy, sad, fearful, disgust, neutral, surprised), then uses Google's Gemini LLM to generate a natural, empathetic response suggestion based on the result.

<br/>

## ✨ Features

- 🎧 **Dual input modes** — upload any audio file (`.wav`, `.m4a`, `.mp3`, etc. via FFmpeg conversion) or record live from your microphone in-browser
- 🧠 **Trained ML model** — Random Forest classifier on MFCC/spectral audio features, honestly benchmarked with 5-fold cross-validation
- 🔬 **Documented experiments** — includes a real feature-engineering comparison (15 vs. 30 features) with results, not just a single untested approach
- 💬 **LLM-powered responses** — chains model output into the Gemini API to generate contextual, empathetic reply suggestions
- 🎨 **Custom-designed UI** — per-emotion color theming, animated transitions, no default templates
- ☁️ **Live deployment** — fully containerized with Docker, deployed on Render

<br/>

## 📊 Results

| Metric | Score |
|---|---|
| Cross-validated accuracy (5-fold) | **54.9% ± 3.2%** |
| Baseline (random chance, 8 classes) | ~12.5% |
| Dataset | RAVDESS (1,440 samples) |

> **Honest limitation:** the model performs reliably on RAVDESS-style scripted/acted speech, but shows the expected accuracy drop on live, casual recordings — a well-known generalization challenge when training on a single acted-emotion dataset. This is documented rather than hidden, since understanding *why* a model underperforms out-of-distribution is as important as the accuracy number itself.

<br/>

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **ML / Data** | Python, scikit-learn, librosa-style feature extraction, NumPy |
| **Backend** | Flask, python-speech-features, FFmpeg / pydub |
| **LLM** | Google Gemini API |
| **Frontend** | HTML5, CSS3, vanilla JavaScript (MediaRecorder API) |
| **DevOps** | Docker, Render, Git/GitHub |

<br/>

## 🚀 Getting Started

```bash
git clone https://github.com/BristiGhosh604/speech-emotion-recognition.git
cd speech-emotion-recognition
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_key_here
```

Run locally:
```bash
python app.py
```

<br/>

## 📁 Project Structure

```
├── app.py                  # Flask app (routes, Gemini integration)
├── Dockerfile               # Container config for deployment
├── config.yaml               # Central settings
├── src/
│   ├── dataset.py             # RAVDESS loader + label parsing
│   ├── features.py             # Audio feature extraction (FFmpeg-based)
│   ├── train.py                 # Model training + cross-validation
├── scripts/
│   └── build_dataset.py          # Feature extraction pipeline
├── templates/
│   └── index.html                 # UI (upload + live recording)
└── outputs/
    └── emotion_model.pkl            # Trained model
```

<br/>

## 📈 Repo Stats

<div align="center">

![GitHub Streak](https://streak-stats.demolab.com?user=BristiGhosh604&theme=default&hide_border=true)

![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=BristiGhosh604&layout=compact&theme=default&hide_border=true&cache_seconds=1800)

</div>

<br/>

## 🗺️ Roadmap

- [ ] Multi-dataset training (CREMA-D, TESS) for better generalization
- [ ] Fine-tuned transformer model (Wav2Vec2) as a modern alternative to classical ML
- [ ] Model card with fairness/robustness analysis across speakers

<br/>

## 📄 License

MIT

<br/>

<div align="center">

**Built by [Bristi Ghosh](https://github.com/BristiGhosh604)**

</div>

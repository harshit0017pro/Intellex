# Intellex — AI Deepfake, Fake News, and Content Credibility Toolkit

Intellex is a hybrid AI system that helps you assess the credibility of text, images, videos, and news articles. It combines local ML models (Hugging Face Transformers) with an optional Gemini API reasoning layer for stronger, explainable verdicts.

- Frontend: Static HTML (Tailwind) app in `Main/` that talks to a local API
- Backend: Python Flask server in `text_detector/app.py`
- Models: Hugging Face models for text and image deepfake detection
- Optional: Gemini API for article analysis and an extra vote on image authenticity

## Features

- Text classification: Human-written vs AI-generated
- Image deepfake detection: Ensemble of two ViT image models + optional Gemini vote
- Video deepfake screening: Sampled frame analysis using the same image pipeline
- News/article URL analysis: Content extraction + Gemini cross-check and summary
- Clean, modern UI in `Main/app.html`

## How it works (high‑level)

- Text: `roberta-base-openai-detector` via Transformers classifies text as Human vs AI.
- Image: Two ViT classifiers vote on deepfakes, optionally combined with a Gemini verdict. Final score = 50% Gemini + 25% model A + 25% model B.
- Video: Samples up to 15 frames and runs the same image pipeline; flags video as deepfake if >50% frames look AI-generated.
- URL: Extracts page text with BeautifulSoup and asks Gemini to: summarize, assign a credibility verdict, list red flags, and surface sources.

Models used (download automatically on first run):
- Text: `roberta-base-openai-detector`
- Image A: `prithivMLmods/Deep-Fake-Detector-v2-Model`
- Image B: `dima806/deepfake_vs_real_image_detection`

## Project structure

```
Intellex/
├─ Main/                    # Frontend (open app.html)
│  ├─ index.html
│  └─ app.html
├─ text_detector/           # Flask backend
│  ├─ app.py                # API server (run me)
│  └─ detector.py           # Text-only detector helper
├─ data/                    # (optional) assets/data
├─ srt/                     # (optional)
├─ test/                    # sample images
│  ├─ fake.jpg
│  └─ real.jpg
└─ requirements.txt
```

## Prerequisites

- Python 3.9+ (3.10 recommended)
- Internet access on first run to download Hugging Face models
- For Gemini features (recommended): a Google AI Studio API key (Gemini)

If PyTorch installation fails via pip, follow the official PyTorch install guide for your OS/CPU/GPU.

## Quick start (Windows PowerShell)

1) Clone and set up a virtual env

```powershell
# From the repo root
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

2) Configure Gemini (optional but required for URL analysis)

- Create a `.env` file at the repo root based on `.env.example`, or set an env var for the session:

```powershell
$env:GEMINI_API_KEY = "your_api_key_here"
```

3) Start the backend (Flask API)

```powershell
python .\text_detector\app.py
```

The API listens on http://127.0.0.1:5000.

4) Open the frontend

- Simple: open `Main/app.html` in your browser, or
- Serve the `Main` folder locally (helps with assets and relative paths):

```powershell
# Option A: Python simple server (serve Main/ at http://127.0.0.1:5500)
python -m http.server 5500 -d .\Main
```

Then browse to http://127.0.0.1:5500/app.html.

## API reference

Base URL: `http://127.0.0.1:5000`

- POST `/detect` — analyze text
  - Body (JSON): `{ "text": "..." }`
  - Returns: `{ type: "text", analysis: "Human Written" | "AI Generated", ai_prob, human_prob }`

- POST `/detect-image` — analyze an image file
  - Body (multipart/form-data): `file: <image>`
  - Returns: `{ type: "image", summary, is_ai, confidence }`

- POST `/detect-video` — analyze a video file (samples up to 15 frames)
  - Body (multipart/form-data): `file: <video>`
  - Returns: `{ type: "video", summary, is_ai, confidence }`

- POST `/detect-url` — analyze an image URL or a news/article URL
  - Body (JSON): `{ "url": "https://..." }`
  - Behavior:
    - If the URL points to an image, runs image analysis and returns the image payload above.
    - Otherwise, extracts article text and asks Gemini to summarize, score credibility, list red flags, and cite sources.
  - Returns (article):
    ```json
    {
      "type": "url_analysis",
      "title": "<the url>",
      "summary": "...",
      "verdict": "Likely Credible | Potentially Misleading | Highly Suspect",
      "red_flags": ["..."],
      "sources": [{"title": "...", "uri": "..."}]
    }
    ```
  - Note: `/detect-url` requires `GEMINI_API_KEY`. Without it, you’ll get an error for article analysis.

## Environment variables

Create a `.env` file in the repo root (or set env vars in your shell):

```
GEMINI_API_KEY=your_api_key_here
```

- Used for: image verdict weighting (50%) and all article/URL analyses
- If missing: image verdict falls back to local models (Gemini counts as neutral), and `/detect-url` article analysis is unavailable

## Troubleshooting

- Torch install issues on Windows
  - Try installing CPU wheels from PyTorch if a generic `pip install torch` fails.
- First request is slow
  - Model weights download on first run; subsequent runs are much faster.
- CORS / network errors from the UI
  - Ensure the Flask server is running on `http://127.0.0.1:5000` and reachable.
  - The frontend hits that origin directly; CORS is enabled in the backend.
- URL analysis fails with an error
  - Ensure `GEMINI_API_KEY` is configured and valid; some responses may be rate-limited.

## Notes and limitations

- Classifiers are probabilistic; use results as decision support, not absolute truth.
- The Gemini API may return rate-limit or transient errors; the backend retries briefly and falls back safely.
- Video analysis samples frames, so extremely short or highly variable videos may be under/overestimated.

## License

This repository is provided as-is for research and educational purposes. Add your preferred license if you plan to distribute.

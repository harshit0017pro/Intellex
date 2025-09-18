import base64
import os
import json
from io import BytesIO
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    ViTImageProcessor,
    ViTForImageClassification,
)
from PIL import Image
import torch
import cv2
import numpy as np
import tempfile
import re
import time 

# --- Setup ---
load_dotenv()
app = Flask(__name__)
CORS(app) 

# --- Load Models ---
print("INFO: Loading all AI models...")
text_tokenizer = RobertaTokenizer.from_pretrained("roberta-base-openai-detector")
text_model = RobertaForSequenceClassification.from_pretrained("roberta-base-openai-detector")
image_processor_1 = ViTImageProcessor.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")
image_model_1 = ViTForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")
image_processor_2 = ViTImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
image_model_2 = ViTForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
print("SUCCESS: All local models loaded.")

# --- API Configuration (Gemini Only) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in .env file.")

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

# --- Reusable Helper Functions ---
def analyze_image_data(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    inputs_1 = image_processor_1(images=image, return_tensors="pt")
    with torch.no_grad():
        ai_prob_1 = torch.nn.functional.softmax(image_model_1(**inputs_1).logits, dim=1)[0][1].item()
    inputs_2 = image_processor_2(images=image, return_tensors="pt")
    with torch.no_grad():
        ai_prob_2 = torch.nn.functional.softmax(image_model_2(**inputs_2).logits, dim=1)[0][0].item()
    gemini_prob = get_gemini_verdict(image_bytes)
    final_ai_prob = (gemini_prob * 0.50) + (ai_prob_1 * 0.25) + (ai_prob_2 * 0.25)
    return final_ai_prob

def get_gemini_verdict(image_bytes):
    if not GEMINI_API_KEY: return 0.5
    
    retries = 3
    delay = 1
    timeout_limit = 30 # Total time limit for the operation in seconds
    start_time = time.time()

    for i in range(retries):
        if time.time() - start_time > timeout_limit:
            print("ERROR: Gemini verdict operation timed out.")
            return 0.5 # Return a neutral score on timeout
            
        try:
            img_base64 = base64.b64encode(image_bytes).decode('utf-8')
            payload = {"contents": [{"parts": [
                {"text": "Is this image a real photograph or is it AI-generated? Answer with only 'Real' or 'AI-Generated'."},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_base64}}
            ]}]}
            response = requests.post(GEMINI_API_URL, json=payload, timeout=20)
            response.raise_for_status()
            text_content = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            if "AI-Generated" in text_content: return 0.99
            elif "Real" in text_content: return 0.01
            return 0.5
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429:
                print(f"WARN: Rate limit hit. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2 
            else:
                print(f"ERROR: Gemini API HTTP error: {http_err}")
                return 0.5
        except Exception as e:
            print(f"ERROR: A general error occurred during the Gemini image check API call: {e}")
            return 0.5
    print("ERROR: Failed to get Gemini verdict after multiple retries.")
    return 0.5


def extract_article_text(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3'])
        article_text = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        if not article_text:
            article_text = soup.get_text(separator=' ', strip=True)
        article_text = re.sub(r'\s+', ' ', article_text)
        return article_text[:4000] + "..." if len(article_text) > 4000 else article_text
    except Exception as e:
        print(f"Error parsing article from {url}: {e}")
        return None

def get_gemini_article_analysis(article_text, url):
    if not GEMINI_API_KEY:
        return {"error": "Gemini analysis is not available (API key is missing)."}
    prompt = f"""
    You are a meticulous fact-checking journalist. Analyze the following news article content from the URL: {url}.
    First, use Google Search to find external information to verify the key claims made in the article.
    Then, provide your analysis as a JSON object with three specific keys: "summary", "verdict", and "red_flags".

    1.  **summary**: A neutral, one-paragraph summary of the article's main points, incorporating any verification you found.
    2.  **verdict**: A single, clear credibility verdict from one of these three options: "Likely Credible", "Potentially Misleading", or "Highly Suspect". Base this on your external verification and the text's quality.
    3.  **red_flags**: A JSON array of strings, listing specific issues found (e.g., "Claims contradict reports from major news outlets," "No named sources for critical data," etc.).

    Here is the article text to analyze:
    ---
    {article_text}
    ---
    Now, provide the analysis as a single, clean JSON object and nothing else.
    """
    
    retries = 3
    delay = 2 
    timeout_limit = 60 # Total time limit for this longer operation
    start_time = time.time()

    for i in range(retries):
        if time.time() - start_time > timeout_limit:
            print("ERROR: Gemini article analysis operation timed out.")
            return {"error": "The analysis took too long to complete. Please try again."}

        try:
            payload = { "contents": [{"parts": [{"text": prompt}]}], "tools": [{"google_search": {}}] }
            response = requests.post(GEMINI_API_URL, json=payload, timeout=45)
            response.raise_for_status()
            json_response = response.json()
            candidate = json_response['candidates'][0]
            response_text = candidate['content']['parts'][0]['text']
            json_str = response_text.strip().lstrip('```json').rstrip('```')
            analysis_result = json.loads(json_str)
            sources = []
            if 'groundingMetadata' in candidate and 'groundingAttributions' in candidate['groundingMetadata']:
                for attribution in candidate['groundingMetadata']['groundingAttributions']:
                    title = attribution['web'].get('title', 'Unknown Title')
                    uri = attribution['web'].get('uri', '#')
                    if uri != '#':
                        sources.append({"title": title, "uri": uri})
            analysis_result['sources'] = sources
            return analysis_result
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429:
                print(f"WARN: Rate limit hit on article analysis. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"ERROR: Gemini API HTTP error: {http_err} - Response: {http_err.response.text}")
                return {"error": "The Gemini API returned an HTTP error during analysis."}
        except Exception as e:
            print(f"ERROR: A general error occurred during Gemini article analysis: {e}")
            return {"error": "Failed to parse the analysis from the AI model."}

    return {"error": "The API is currently busy. Please try again in a moment."}


# --- API Routes ---
@app.route("/detect", methods=["POST"])
def detect_text():
    try:
        data = request.json
        text = data.get("text", "")
        if not text.strip(): return jsonify({"error": "No text provided"}), 400
        inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = text_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            ai_prob = probs[0][1].item()
            human_prob = probs[0][0].item()
        result = { "type": "text", "analysis": "AI Generated" if ai_prob > human_prob else "Human Written", "ai_prob": round(ai_prob * 100, 2), "human_prob": round(human_prob * 100, 2) }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": "Server error during text analysis."}), 500

@app.route("/detect-image", methods=["POST"])
def detect_image():
    try:
        if "file" not in request.files: return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        final_ai_prob = analyze_image_data(file.read())
        is_ai = final_ai_prob > 0.5
        confidence = final_ai_prob * 100 if is_ai else (1 - final_ai_prob) * 100
        verdict = "likely AI-generated" if is_ai else "likely a real photograph"
        summary = f"Based on a hybrid analysis, this image is {verdict} with a confidence of {round(confidence, 2)}%."
        result = { "type": "image", "summary": summary, "is_ai": is_ai, "confidence": round(confidence, 2)}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": "Server error during image analysis."}), 500

@app.route("/detect-video", methods=["POST"])
def detect_video():
    try:
        if "file" not in request.files: return jsonify({"error": "No file uploaded"}), 400
        video_file = request.files["file"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            video_file.save(tmp.name)
            tmp_video_path = tmp.name
        cap = cv2.VideoCapture(tmp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: 
            cap.release(); os.unlink(tmp_video_path)
            return jsonify({"error": "Could not read video file."}), 400
        num_frames_to_analyze = min(15, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, num_frames_to_analyze, dtype=int)
        ai_frame_count = 0
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_ai_prob = analyze_image_data(buffer.tobytes())
                if frame_ai_prob > 0.5: ai_frame_count += 1
        cap.release(); os.unlink(tmp_video_path)
        ai_percentage = (ai_frame_count / num_frames_to_analyze) * 100
        is_deepfake = ai_percentage > 50
        verdict = "likely a Deepfake" if is_deepfake else "likely a real video"
        summary = f"Based on an analysis of {num_frames_to_analyze} frames, this video is {verdict}. {ai_frame_count} frames were flagged as potentially AI-generated ({round(ai_percentage, 1)}%)."
        result = { "type": "video", "summary": summary, "is_ai": is_deepfake, "confidence": ai_percentage if is_deepfake else 100 - ai_percentage}
        return jsonify(result)
    except Exception as e:
        print(f"ERROR in /detect-video: {e}")
        return jsonify({"error": "An error occurred during video analysis."}), 500

@app.route("/detect-url", methods=["POST"])
def detect_url():
    print("\nINFO: Received request for /detect-url")
    try:
        data = request.json
        url = data.get("url", "").strip()
        if not url: return jsonify({"error": "No URL provided"}), 400
        if not re.match(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', url):
            return jsonify({"error": "Invalid URL format."}), 400

        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
        if url.lower().endswith(image_extensions):
            response = requests.get(url, stream=True, timeout=15)
            response.raise_for_status()
            final_ai_prob = analyze_image_data(response.content)
            is_ai = final_ai_prob > 0.5
            confidence = final_ai_prob * 100 if is_ai else (1 - final_ai_prob) * 100
            summary = f"Based on a hybrid analysis, this image is {'likely AI-generated' if is_ai else 'likely a real photograph'} with a confidence of {round(confidence, 2)}%."
            result = { "type": "image", "summary": summary, "is_ai": is_ai, "confidence": round(confidence, 2), "source_url": url }
            return jsonify(result)
        else:
            article_text = extract_article_text(url)
            if not article_text or len(article_text) < 100:
                return jsonify({"error": "Could not extract enough text from the URL."}), 400
            
            analysis = get_gemini_article_analysis(article_text, url)
            
            if "error" in analysis:
                return jsonify({"error": analysis["error"]}), 500

            result = {
                "type": "url_analysis",
                "title": url,
                "summary": analysis.get("summary", "No summary available."),
                "verdict": analysis.get("verdict", "Unknown"),
                "red_flags": analysis.get("red_flags", []),
                "sources": analysis.get("sources", [])
            }
            return jsonify(result)

    except Exception as e:
        print(f"ERROR in /detect-url: {e}")
        return jsonify({"error": "A critical error occurred during URL analysis."}), 500

# --- Run Application ---
if __name__ == "__main__":
    print("INFO: Starting Flask server...")
    app.run(port=5000, debug=True, use_reloader=False)


# app.py
import os
import json
import re
import logging
from functools import lru_cache
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Cache language data to avoid re-reading files on every request
@lru_cache(maxsize=32)
def load_language_data(lang_id):
    """Loads language data from a JSON file and caches it."""
    lang_file_path = os.path.join(
        os.path.dirname(__file__), "languages", f"{lang_id}.json"
    )
    if not os.path.exists(lang_file_path):
        logger.warning(f"Language file not found: {lang_file_path}")
        return None
    with open(lang_file_path, "r", encoding="utf-8") as f:
        logger.info(f"Successfully loaded language data for: {lang_id}")
        return json.load(f)

def clean_punctuation(text):
    """Cleans up punctuation and whitespace in a string."""
    if not text:
        return text
    
    cleaned_text = text
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    cleaned_text = re.sub(r"([.,;:!?])[.,;:!?]+", r"\1", cleaned_text)
    cleaned_text = re.sub(r"\s+([.,;:!?])", r"\1", cleaned_text)
    cleaned_text = re.sub(r"\s+([\")])", r"\1", cleaned_text)
    cleaned_text = re.sub(r"([([`\"])\s+", r"\1", cleaned_text)
    cleaned_text = re.sub(r"\.{2,}", "...", cleaned_text)
    cleaned_text = re.sub(r"(\r\n|\n|\r){2,}", "\n", cleaned_text)
    return cleaned_text.strip()

def filter_filler_words(text, phrases):
    """Removes filler words from text using a list of phrases, handling punctuation."""
    if not text or not phrases:
        return text
    
    cleaned_text = text
    sorted_phrases = sorted(phrases, key=len, reverse=True)
    
    for phrase in sorted_phrases:
        if not phrase or len(phrase) < 2:
            continue
        
        escaped_phrase = re.escape(phrase)
        # Use a more flexible regex that matches the phrase surrounded by word boundaries or punctuation.
        # This regex looks for the phrase as a whole word, optionally followed or preceded by punctuation.
        # \b ensures it's a whole word, and [.,;:!?]? handles optional trailing punctuation.
        regex = re.compile(
            rf"\b({escaped_phrase})\b[.,;:!?]?",
            re.IGNORECASE | re.UNICODE,
        )
        
        match = regex.search(cleaned_text)
        if match:
            logger.info(f"Removed phrase '{phrase}'")
            # Replace the match with a single space.
            cleaned_text = regex.sub(" ", cleaned_text)
    
    return re.sub(r'\s+', ' ', cleaned_text).strip()

def detect_languages(text, languages_data):
    """Detects languages in a given text based on character sets."""
    if not text:
        return []
    
    detected = []
    for lang_id, lang_data in languages_data.items():
        if not lang_data or not lang_data.get("detection"):
            continue
        
        detection = lang_data["detection"]
        char_set = set(detection.get("characters", "").lower())
        threshold = detection.get("threshold", 0.3)
        
        match_count = 0
        letter_count = 0
        matched_chars = set()
        
        for char in text:
            char = char.lower()
            if char.isalpha():
                letter_count += 1
                if char in char_set:
                    match_count += 1
                    matched_chars.add(char)
                    
        if letter_count > 0 and (match_count / letter_count) >= threshold and len(matched_chars) >= 3:
            detected.append(lang_id)
            logger.info(f"Detected language: {lang_id}")
            
    return detected

@app.route("/clean", methods=["POST"])
def clean_text_service():
    """API endpoint to clean text based on language rules."""
    data = request.json
    prompt = data.get("prompt", "")
    
    if not prompt:
        logger.warning("Received an empty prompt.")
        return jsonify({"cleanedPrompt": ""})
    
    all_languages = {
        "english": load_language_data("english"),
    }
    
    detected_languages = detect_languages(prompt, all_languages)
    
    if not detected_languages:
        logger.info("No language detected, defaulting to English.")
        detected_languages.append("english")
    
    cleaned_prompt = prompt
    for lang_id in detected_languages:
        lang_data = all_languages.get(lang_id)
        if lang_data:
            phrases_to_remove = lang_data.get("phrases", [])
            cleaned_prompt = filter_filler_words(cleaned_prompt, phrases_to_remove)
            
    cleaned_prompt = clean_punctuation(cleaned_prompt)
    logger.info(f"Original prompt: '{prompt}'")
    logger.info(f"Cleaned prompt: '{cleaned_prompt}'")
    
    return jsonify({"cleanedPrompt": cleaned_prompt})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

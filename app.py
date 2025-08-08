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
    cleaned_text = re.sub(r"([.,;:!?])\s*\1+", r"\1", cleaned_text)
    cleaned_text = re.sub(r"\s+([.,;:!?])", r"\1", cleaned_text)
    cleaned_text = re.sub(r"\s+([\")])", r"\1", cleaned_text)
    cleaned_text = re.sub(r"([([`\"])\s+", r"\1", cleaned_text)
    cleaned_text = re.sub(r"\.{2,}", "...", cleaned_text)
    cleaned_text = re.sub(r"(\r\n|\n|\r){2,}", "\n", cleaned_text)
    return cleaned_text.strip()

def process_text_with_rules(text, rules):
    """Applies a list of rules (remove, replace, with stop phrases) to a text."""
    cleaned_text = text
    
    for rule in rules:
        action = rule.get("action")
        phrases = rule.get("phrases", [])
        with_text = rule.get("with", "")
        stop_phrases = rule.get("stop_phrases", [])
        
        # Sort phrases by length to prevent partial replacements
        sorted_phrases = sorted(phrases, key=len, reverse=True)
        
        # Pre-compile stop phrase regexes for efficiency
        stop_regexes = []
        for p in stop_phrases:
            stop_regexes.append(re.compile(r'\b' + re.escape(p) + r'\b', re.IGNORECASE | re.UNICODE))

        for phrase in sorted_phrases:
            if not phrase or len(phrase) < 2:
                continue

            # Check for stop phrases first
            should_stop = False
            for stop_regex in stop_regexes:
                if stop_regex.search(cleaned_text):
                    should_stop = True
                    logger.info(f"Stop phrase '{stop_regex.pattern}' found. Skipping action for '{phrase}'.")
                    break
            
            if should_stop:
                continue

            escaped_phrase = re.escape(phrase)
            # Match the whole word, ignoring surrounding punctuation
            regex = re.compile(
                rf'\b{escaped_phrase}\b',
                re.IGNORECASE | re.UNICODE
            )
            
            if action == "remove":
                if regex.search(cleaned_text):
                    logger.info(f"Removed phrase '{phrase}'")
                    cleaned_text = regex.sub(' ', cleaned_text)
            elif action == "replace":
                if regex.search(cleaned_text):
                    logger.info(f"Replaced phrase '{phrase}' with '{with_text}'")
                    cleaned_text = regex.sub(with_text, cleaned_text)
    
    return cleaned_text

@app.route("/clean", methods=["POST"])
def clean_text_service():
    """API endpoint to clean text based on language and model rules."""
    data = request.json
    prompt = data.get("prompt", "")
    model = data.get("model", "default")
    
    if not prompt:
        logger.warning("Received an empty prompt.")
        return jsonify({"cleanedPrompt": ""})
    
    lang_id = f"english_{model.lower()}"
    lang_data = load_language_data(lang_id)
    
    if not lang_data:
        # Fallback to a generic English file if model-specific file is not found
        lang_id = "english_default"
        lang_data = load_language_data(lang_id)
        if not lang_data:
            logger.warning(f"No language data found for model '{model}' or default. Returning original prompt.")
            return jsonify({"cleanedPrompt": prompt})

    logger.info(f"Using language rules from: {lang_id}")
    
    cleaned_prompt = prompt
    rules = lang_data.get("rules", [])
    
    # Process the text using the rules from the loaded language file
    cleaned_prompt = process_text_with_rules(cleaned_prompt, rules)
    
    # Final punctuation and spacing cleanup
    cleaned_prompt = clean_punctuation(cleaned_prompt)
    
    logger.info(f"Original prompt: '{prompt}'")
    logger.info(f"Cleaned prompt: '{cleaned_prompt}'")
    
    return jsonify({"cleanedPrompt": cleaned_prompt})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

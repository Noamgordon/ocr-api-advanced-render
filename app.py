# app.py
import os
import json
import re
import logging
import jwt
import psycopg2
from functools import lru_cache, wraps
from flask import Flask, request, jsonify, g

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load the JWT secret key from environment variables.
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')
if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY environment variable not set. This is required for API security.")

# Load database credentials from environment variables
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_PORT = os.environ.get("DB_PORT", "5432")

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

# New function to check the API key against the database
def verify_api_key_in_db(api_key):
    try:
        # We need a new connection for each thread in Gunicorn
        conn = psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            sslmode='require' # Supabase requires SSL
        )
        cur = conn.cursor()
        
        # The key is hashed inside the database for verification.
        # This is a secure way to compare without exposing the key.
        cur.execute("""
            SELECT user_id, is_active FROM api_keys
            WHERE crypt(%s, key_hash) = key_hash AND is_active = TRUE;
        """, (api_key,))
        
        result = cur.fetchone()
        
        # If a valid key is found, update the last_used_at timestamp.
        if result:
            user_id = result[0]
            cur.execute("""
                UPDATE api_keys SET last_used_at = NOW() WHERE crypt(%s, key_hash) = key_hash;
            """, (api_key,))
            conn.commit()
            return user_id
        
    except psycopg2.Error as e:
        logger.error(f"Database error during key verification: {e}")
        # In case of an error, assume the key is invalid
        return None
    finally:
        if conn:
            cur.close()
            conn.close()
    return None

# New decorator to handle both JWT and custom API keys
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        user_id = None

        if not auth_header:
            return jsonify({'message': 'Authorization header is missing.'}), 401
        
        try:
            # Check for a standard Bearer JWT token from Supabase
            if auth_header.startswith('Bearer '):
                token = auth_header.split(" ")[1]
                data = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
                user_id = data.get('sub')
                if not user_id:
                    return jsonify({'message': 'Invalid JWT token.'}), 401
                logger.info(f"User authenticated with JWT: {user_id}")

            # Check for a custom API Key
            elif auth_header.startswith('sk_'):
                api_key = auth_header
                user_id = verify_api_key_in_db(api_key)
                if not user_id:
                    return jsonify({'message': 'Invalid or inactive API key.'}), 401
                logger.info(f"User authenticated with API key: {user_id}")

            else:
                return jsonify({'message': 'Authorization header format is invalid.'}), 401

            g.user = user_id
            
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return jsonify({'message': 'Token has expired or is invalid!'}), 401
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return jsonify({'message': f'Authentication failed: {e}'}), 401
        
        return f(*args, **kwargs)
    return decorated

@app.route("/clean", methods=["POST"])
@token_required
def clean_text_service():
    """API endpoint to clean text based on language and model rules, with API key authentication."""
    data = request.json
    prompt = data.get("prompt", "")
    model = data.get("model", "default")
    
    if not prompt:
        logger.warning("Received an empty prompt.")
        return jsonify({"cleanedPrompt": ""})
    
    logger.info(f"Request from user '{g.user}'.")
    
    lang_id = f"english_{model.lower()}"
    lang_data = load_language_data(lang_id)
    
    if not lang_data:
        lang_id = "english_default"
        lang_data = load_language_data(lang_id)
        if not lang_data:
            logger.warning(f"No language data found for model '{model}' or default. Returning original prompt.")
            return jsonify({"cleanedPrompt": prompt})

    logger.info(f"Using language rules from: {lang_id}")
    
    cleaned_prompt = prompt
    rules = lang_data.get("rules", [])
    
    cleaned_prompt = process_text_with_rules(cleaned_prompt, rules)
    cleaned_prompt = clean_punctuation(cleaned_prompt)
    
    logger.info(f"Original prompt: '{prompt}'")
    logger.info(f"Cleaned prompt: '{cleaned_prompt}'")
    
    return jsonify({"cleanedPrompt": cleaned_prompt})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

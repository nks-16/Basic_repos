import os
import base64
import requests
import json
import logging
import re
from pathlib import Path

# === Configuration ===
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-04-17:generateContent"
)

PROMPT_TEXT = """You are an expert UI analyst. Your task is to analyze user-uploaded screenshots of web or app designs (e.g. Figma screens) and identify all visible user interface elements in the image.

First, identify what type of page it is (e.g., login_page, signup_page, homepage, booking_page, etc.).

Return the page type in this exact format on the first line:
# Page type: <snake_case_page_type_name>

Then, for each identified UI element, provide output in the following format:

@abstractmethod
def [element_name](self):
    '''[Short, clear description of the element’s purpose or label in the UI]'''

[element_name] should be a concise, snake_case name describing the element’s role, e.g. src, login_button, username_field, trip_type_dropdown.

The description should explain what the element does, what is the purpose and intention of the web element.

Include every unique UI element you detect: buttons, labels, icons, text fields, dropdowns, radio buttons, toggles, checkboxes, images, etc.

Ignore decorative or purely stylistic elements unless they have a clear interactive or informational purpose.

Keep each description short and factual.

Present all outputs as a single list of abstract methods in code format."""

# === Path Configuration ===
BASE_DIR = Path(__file__).parent.resolve()
IMAGE_FOLDER = BASE_DIR / "images"
OUTPUT_FOLDER = BASE_DIR / "description"
TEMPERATURE = 0.0

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === Headers ===
HEADERS = {
    "Content-Type": "application/json",
    "x-goog-api-key": API_KEY
}

# === Helper Functions ===
def encode_image_to_base64(filepath):
    try:
        with open(filepath, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"Failed to encode {filepath}: {e}")
        return None

def build_payload(base64_image_str):
    return {
        "system_instruction": {
            "parts": [{"text": PROMPT_TEXT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64_image_str
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": TEMPERATURE
        }
    }

def send_request(payload):
    try:
        response = requests.post(
            GEMINI_API_URL,
            headers=HEADERS,
            data=json.dumps(payload),
            timeout=30
        )
        response.raise_for_status()
        json_response = response.json()
        return json_response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    except requests.exceptions.Timeout:
        logging.warning("Request timed out.")
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in response: {e}")
    return None

def extract_page_type(text):
    match = re.search(r"#\s*Page type:\s*([a-z_]+)", text)
    return match.group(1) if match else "unknown_page"


def get_unique_filename(folder, base_name, ext):
    counter = 1
    candidate = f"{base_name}{ext}"
    while (folder / candidate).exists():
        candidate = f"{base_name}_{counter}{ext}"
        counter += 1
    return candidate

def save_output(filename, content):
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_FOLDER / filename
    try:
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(content)
        logging.info(f"Saved output to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save output for {filename}: {e}")

def process_images():
    if not IMAGE_FOLDER.exists():
        logging.error(f"Image folder not found: {IMAGE_FOLDER}")
        return

    images = [f for f in IMAGE_FOLDER.iterdir() if f.suffix.lower() == ".png"]
    if not images:
        logging.warning("No PNG images found in the folder.")
        return

    for img_path in images:
        base64_str = encode_image_to_base64(img_path)
        if not base64_str:
            continue

        payload = build_payload(base64_str)
        response_text = send_request(payload)

        if response_text:
            page_type = extract_page_type(response_text)
            new_img_name = get_unique_filename(IMAGE_FOLDER, page_type, ".png")
            new_txt_name = Path(new_img_name).with_suffix(".txt").name

            try:
                new_img_path = IMAGE_FOLDER / new_img_name
                img_path.rename(new_img_path)
                logging.info(f"Renamed image: {img_path.name} → {new_img_name}")
            except Exception as e:
                logging.error(f"Failed to rename image {img_path.name}: {e}")
                continue

            save_output(new_txt_name, response_text)
        else:
            logging.warning(f"No response received for {img_path.name}")

# === Entry Point ===
if __name__ == "__main__":
    if not API_KEY:
        logging.error("❌ API key is missing or not set.")
    else:
        logging.info("🚀 Starting UI analysis pipeline...")
        process_images()
        logging.info("✅ Pipeline complete.")

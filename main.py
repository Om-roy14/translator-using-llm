from dotenv import load_dotenv
import os
import requests

# Load GROQ_API_KEY from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"

def build_translation_prompt(text, src_lang, tgt_lang):
    if src_lang:
        prompt = (
            f"You are a translation assistant. Translate the following text from {src_lang} to {tgt_lang}:\n"
            f"\"{text}\"\n"
            "Only provide the translated text, no explanations."
        )
    else:
        prompt = (
            f"You are a translation assistant. Detect the language of this text and translate it to {tgt_lang}:\n"
            f"\"{text}\"\n"
            "Only provide the translated text, no explanations."
        )
    return prompt

def translate_with_groq(text, tgt_lang="English", src_lang=None):
    prompt = build_translation_prompt(text, src_lang, tgt_lang)
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful AI translator."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1000
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    return result['choices'][0]['message']['content'].strip()

if __name__ == "__main__":
    print("Groq Llama-3 Translator (any language <-> English or between two languages)")
    print("Leave Source Language blank to auto-detect language.")
    print("Enter 'quit' to exit.\n")

    while True:
        src_lang = input("Source language (e.g. French, [auto-detect] if blank): ").strip()
        tgt_lang = input("Target language (e.g. English, Hindi): ").strip()
        text = input("Text to translate: ").strip()

        if text.lower() in ("quit", "exit"):
            break

        try:
            translation = translate_with_groq(text, tgt_lang, src_lang if src_lang else None)
            print(f"\nTranslation: {translation}\n")
        except Exception as e:
            print(f"Error: {e}\n")

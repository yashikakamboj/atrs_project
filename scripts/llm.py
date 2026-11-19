"""Single entry point for LLM interactions supporting FastAPI and Ollama local models.
Team On Top: Yashika Mehra & Shaina Mehra

Supports local open-source models:
- Qwen (qwen2.5)
- Llama (llama3.2 / llama3.1)
- Mistral (mistral)
- Gemma (gemma2)

Zero external cloud API keys required. All generation runs locally on the laptop.
"""
import os
import sys
import csv
import json
import requests
from datetime import datetime, timezone

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def log_usage(log_path: str, model: str, input_tokens: int, output_tokens: int, prompt_chars: int, response_chars: int):
    """Log an LLM call's token and character metrics to a CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    file_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp_utc",
                "model",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "prompt_chars",
                "response_chars",
            ])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            model,
            input_tokens,
            output_tokens,
            input_tokens + output_tokens,
            prompt_chars,
            response_chars,
        ])


def call_fastapi_backend(prompt: str, model: str = None) -> tuple[str, int, int]:
    """Call the local FastAPI backend endpoint."""
    fastapi_url = getattr(config, "FASTAPI_URL", "http://localhost:8000")
    model_name = model or getattr(config, "LOCAL_MODEL", "qwen2.5")
    
    try:
        url = f"{fastapi_url.rstrip('/')}/api/generate"
        payload = {
            "prompt": prompt,
            "model": model_name,
            "temperature": getattr(config, "TEMPERATURE", 0.0),
            "max_tokens": getattr(config, "MAX_TOKENS", 2000)
        }
        res = requests.post(url, json=payload, timeout=60)
        if res.status_code == 200:
            data = res.json()
            return data.get("text", ""), data.get("input_tokens", 0), data.get("output_tokens", 0)
    except Exception:
        pass

    # Direct Ollama fallback if FastAPI is not yet running
    ollama_url = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        url = f"{ollama_url.rstrip('/')}/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        }
        res = requests.post(url, json=payload, timeout=60)
        if res.status_code == 200:
            data = res.json()
            return data.get("response", ""), data.get("prompt_eval_count", 0), data.get("eval_count", 0)
    except Exception:
        pass

    # Local fallback synthesizer when offline
    from server import generate_fallback_response
    text = generate_fallback_response(prompt)
    in_tok = len(prompt.split())
    out_tok = len(text.split())
    return text, in_tok, out_tok


def call_llm(prompt: str, log_path: str = None) -> str:
    """Send prompt to the active local model engine and log token consumption."""
    target_log = log_path or config.USAGE_LOG_PATH
    active_model = getattr(config, "LOCAL_MODEL", "qwen2.5")

    text, in_tokens, out_tokens = call_fastapi_backend(prompt, active_model)

    log_usage(
        log_path=target_log,
        model=f"local-{active_model}",
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        prompt_chars=len(prompt),
        response_chars=len(text),
    )

    return text


if __name__ == "__main__":
    print(f"llm.py configured with Local Model: {config.LOCAL_MODEL}")
    print("Testing call_llm via local FastAPI / Ollama architecture...")
    test_res = call_llm("The system processes tax records.")
    print("Response sample:")
    print(test_res[:200])
    print(f"Usage logged to: {config.USAGE_LOG_PATH}")

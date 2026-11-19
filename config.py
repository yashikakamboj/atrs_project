"""Central configuration settings for the ATRS requirements study.
Team On Top: Yashika Mehra & Shaina Mehra

Local AI Architecture using FastAPI & Ollama:
- Qwen (qwen2.5)
- Llama (llama3.2 / llama3.1)
- Mistral (mistral)
- Gemma (gemma2)

Zero external cloud API keys required. 100% private and runs locally on any laptop.
"""
import os
import sys

# ------------------------------------------------------------
# Automatic .env File Loader
# ------------------------------------------------------------
def _load_env_file(env_path: str = None) -> None:
    """Load key-value pairs from .env into os.environ if not already present."""
    if env_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(base_dir, ".env")
    
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("\"'").strip()
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

_load_env_file()

# Base paths
DATA_DIR = os.getenv("ATRS_DATA_DIR", "data")
OUT_DIR = os.getenv("ATRS_OUT_DIR", "outputs")
TEXTS_DIR = os.path.join(DATA_DIR, "texts")
USAGE_LOG_PATH = os.path.join(DATA_DIR, "usage_log.csv")

# Reproducibility seed
SEED = 42

# Sampling and dataset parameters
N_CANDIDATES = 70             # Records sampled from the ATRS metadata index
N_FINAL = 40                  # Final sample size evenly spaced across completeness (proposal target: 30-50)
MIN_WORDS = 300               # Drop records with fewer words than this threshold

# Local AI Engine Settings (FastAPI + Ollama)
PROVIDER = os.getenv("LLM_PROVIDER", "fastapi")
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Active local model: 'qwen2.5', 'llama3.2', 'mistral', 'gemma2'
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "qwen2.5")
MODEL = LOCAL_MODEL           # Alias for compatibility

TEMPERATURE = 0.0             # Deterministic generation
MAX_TOKENS = 4000
MAX_SOURCE_CHARS = 60000      # Truncation cap for very long source texts to prevent context overflow

# Evaluation and annotation settings
DOUBLE_ANNOTATE_FRACTION = 0.20  # 20% subset scored by Rater B for inter-rater agreement (Cohen's kappa)

"""
PricePulse — Global Settings & Configuration
=============================================
Loads environment variables and provides configuration constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Load .env file ───
# config/ is one level down from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')


class Settings:
    """Application settings loaded from environment."""

    # ── Gemini API ──
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
    GEMINI_MODEL: str = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')

    # ── Paths ──
    BASE_DIR: Path = BASE_DIR
    MODEL_PATH: Path = BASE_DIR / 'models' / 'model_price_predictor.pkl'
    TFIDF_PATH: Path = BASE_DIR / 'models' / 'tfidf_vectorizer.pkl'
    METADATA_PATH: Path = BASE_DIR / 'models' / 'model_metadata.json'

    # ── LLM Config ──
    LLM_TEMPERATURE: float = 0.3       # Low for consistent pricing
    LLM_MAX_TOKENS: int = 1024
    LLM_MAX_RETRIES: int = 3
    LLM_TIMEOUT: int = 30              # seconds

    # ── Pricing Config ──
    MIN_PRICE: float = 0.99
    MAX_PRICE: float = 99999.99
    DEFAULT_CURRENCY: str = 'USD'

    @classmethod
    def is_llm_configured(cls) -> bool:
        """Check if the LLM API key is set."""
        return bool(cls.GEMINI_API_KEY) and cls.GEMINI_API_KEY != 'your_gemini_api_key_here'

    @classmethod
    def validate(cls) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings_list = []
        if not cls.is_llm_configured():
            warnings_list.append("⚠️  GEMINI_API_KEY not set. LLM features disabled (ML-only mode).")
        if not cls.MODEL_PATH.exists():
            warnings_list.append(f"⚠️  Model not found: {cls.MODEL_PATH}")
        if not cls.METADATA_PATH.exists():
            warnings_list.append(f"⚠️  Metadata not found: {cls.METADATA_PATH}")
        return warnings_list

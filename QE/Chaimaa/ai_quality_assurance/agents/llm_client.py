"""
PricePulse — Gemini LLM Client
================================
Wrapper around the Google Gen AI SDK (google-genai) with:
  - JSON mode for structured responses
  - Retry logic with exponential backoff
  - Response validation
  - Token usage tracking
"""

import json
import time
from google import genai
from google.genai import types
from config.settings import Settings


class GeminiClient:
    """
    Google Gemini API client for PricePulse.

    Handles:
      - Model initialization & configuration
      - JSON-structured generation
      - Error handling & retries
      - Token usage tracking
    """

    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize the Gemini client.

        Args:
            api_key: Gemini API key (defaults to Settings)
            model_name: Model to use (defaults to Settings)
        """
        self.api_key = api_key or Settings.GEMINI_API_KEY
        self.model_name = model_name or Settings.GEMINI_MODEL
        self.total_tokens_used = 0
        self.total_requests = 0

        if not self.api_key or self.api_key == 'your_gemini_api_key_here':
            raise ValueError(
                "Gemini API key not configured. "
                "Set GEMINI_API_KEY in your .env file.\n"
                "Get a key at: https://aistudio.google.com/apikey"
            )

        # Create the client
        self.client = genai.Client(api_key=self.api_key)
        self._initialized = True

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = None,
    ) -> dict:
        """
        Generate a JSON response from the LLM.

        Args:
            system_prompt: System instruction for the model
            user_prompt: User message/query
            temperature: Override temperature (optional)

        Returns:
            Parsed JSON dict from the LLM response

        Raises:
            ValueError: If JSON parsing fails after retries
            Exception: If API call fails after retries
        """
        temp = temperature if temperature is not None else Settings.LLM_TEMPERATURE

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temp,
            max_output_tokens=Settings.LLM_MAX_TOKENS,
            response_mime_type="application/json",
        )

        last_error = None
        for attempt in range(Settings.LLM_MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=config,
                )
                self.total_requests += 1

                # Track token usage
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    self.total_tokens_used += (
                        getattr(response.usage_metadata, 'total_token_count', 0)
                    )

                # Extract and parse JSON
                text = response.text.strip()

                # Clean up potential markdown wrapping
                if text.startswith('```json'):
                    text = text[7:]
                if text.startswith('```'):
                    text = text[3:]
                if text.endswith('```'):
                    text = text[:-3]
                text = text.strip()

                result = json.loads(text)
                return result

            except json.JSONDecodeError as e:
                last_error = e
                if attempt < Settings.LLM_MAX_RETRIES - 1:
                    time.sleep(1 * (attempt + 1))  # Backoff
                    continue

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Rate limit — wait and retry
                if 'rate' in error_str or '429' in error_str:
                    wait_time = 5 * (attempt + 1)
                    time.sleep(wait_time)
                    continue

                # Other API errors
                if attempt < Settings.LLM_MAX_RETRIES - 1:
                    time.sleep(2 * (attempt + 1))
                    continue

        raise ValueError(
            f"Failed to get valid JSON after {Settings.LLM_MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    def get_usage_stats(self) -> dict:
        """Return token usage statistics."""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens_used,
            "model": self.model_name,
        }

    def health_check(self) -> bool:
        """Quick test to verify the API connection works."""
        try:
            config = types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=50,
                response_mime_type="application/json",
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents='Respond with: {"status": "ok"}',
                config=config,
            )
            result = json.loads(response.text.strip())
            return result.get("status") == "ok"
        except Exception:
            return False

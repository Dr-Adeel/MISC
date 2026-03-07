import pytest
import os


@pytest.fixture(autouse=True)
def clear_scraper_env_vars(monkeypatch):
    """
    Clear external service environment variables for tests.
    This ensures tests use mock implementations instead of real API endpoints.
    """
    monkeypatch.delenv("SCRAPER_API_KEY", raising=False)
    monkeypatch.delenv("ANALYZER_API_URL", raising=False)
    monkeypatch.delenv("AI_AGENT_API_URL", raising=False)

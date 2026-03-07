import pytest
from unittest.mock import AsyncMock
from app.controller import PriceController
from app.schemas import ProductRequest

@pytest.mark.asyncio
async def test_price_controller_workflow():

    mock_scraper = AsyncMock()
    mock_analyzer = AsyncMock()
    mock_ai_agent = AsyncMock()

    mock_scraper.fetch_listings.return_value = [
        {"title": "iPhone 13 - Listing 1", "price": 100, "source": "amazon"},
        {"title": "iPhone 13 - Listing 2", "price": 120, "source": "ebay"},
        {"title": "iPhone 13 - Listing 3", "price": 110, "source": "amazon"}
    ]

    mock_analyzer.calculate_stats.return_value = {
        "mean": 110,
        "median": 110,
        "min": 100,
        "max": 120
    }

    mock_ai_agent.get_price_advice.return_value = {
        "price": 115,
        "confidence": 0.85,
        "justification": "Competitive market price"
    }

    controller = PriceController(
        scraper=mock_scraper,
        analyzer=mock_analyzer,
        ai_agent=mock_ai_agent
    )

    request = ProductRequest(product_name="iPhone 13")

    result = await controller.process_request(request)

    assert result.recommended_price == 115
    assert result.confidence_score == 0.85
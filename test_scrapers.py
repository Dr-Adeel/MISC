import pytest
from app.providers.ebay_provider import EbayProvider
from app.exceptions import MarketUnreachableError
from app.providers.amazon_provider import AmazonProvider

@pytest.mark.asyncio
async def test_ebay_fetch_listings():

    provider = EbayProvider()

    listings = await provider.fetch_listings("iphone 13")

    assert isinstance(listings, list)

@pytest.mark.asyncio
async def test_market_unreachable():

    provider = AmazonProvider()

    with pytest.raises(MarketUnreachableError):
        await provider.fetch_listings("invalid product")
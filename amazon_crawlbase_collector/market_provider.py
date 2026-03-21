from abc import ABC, abstractmethod
from typing import List, Dict, Any

class MarketProvider(ABC):
    """
    Abstract Base Class for market data providers.
    Defines the contract that all concrete providers must implement.
    """

    def __init__(self, provider_name: str):
        """
        Initialize the market provider.
        
        Args:
            provider_name (str): Name of the market provider (e.g., 'eBay', 'Amazon')
        """
        self.provider_name = provider_name

    @abstractmethod
    def fetch_listings(self, product_name: str, domain: str) -> List[Dict[str, Any]]:
        """
        Fetch listings for a given product from the market.
        
        Args:
            product_name (str): The product name to search for
            domain (str): The domain to search on (e.g. amazon.com)
            
        Returns:
            List[Dict[str, Any]]: List of raw listing data
        """
        pass

    @abstractmethod
    def clean_data(self, raw_listings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean and standardize the raw listing data.
        
        Args:
            raw_listings (List[Dict[str, Any]]): Raw data from market API/scraper
            
        Returns:
            List[Dict[str, Any]]: Cleaned and standardized listing data
        """
        pass

    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return self.provider_name

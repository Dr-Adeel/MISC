import requests
import time
import urllib.parse
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CrawlbaseClient:
    """
    Client for the Crawlbase API (formerly ProxyCrawl) to scrape Amazon product data.
    """
    
    BASE_URL = "https://api.crawlbase.com/"

    def __init__(self, token: str):
        if not token:
            raise ValueError("Crawlbase Token is missing. Please check your .env file.")
        self.token = token

    def get_products(self, domain: str, search_term: str, page: int = 1) -> List[Dict[str, Any]]:
        """
        Fetch products from Amazon using Crawlbase 'amazon-serp' scraper.
        
        Args:
            domain (str): Amazon domain (e.g., 'amazon.com')
            search_term (str): Keyword to search for
            page (int): Page number to scrape (default: 1)
            
        Returns:
            List[Dict]: List of found products
        """
        # Construct the URL to scrape
        # Example: https://www.amazon.com/s?k=laptop&page=2
        amazon_url = f"https://www.{domain}/s?k={urllib.parse.quote(search_term)}&page={page}"
        
        params = {
            'token': self.token,
            'scraper': 'amazon-serp', # Explicitly use the Amazon SERP scraper
            'url': amazon_url,
            'format': 'json', # Explicitly request JSON
        }

        logger.info(f"Fetching '{search_term}' from {domain}...")
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=60)
            
            if response.status_code != 200:
                logger.error(f"API Error ({response.status_code}): {response.text}")
                return []
                
            data = response.json()
            
            # The structure of Crawlbase scraper response depends on the scraper
            # Usually data['body'] contains the products list
            extracted_body = data.get('body', {})
            
            if not extracted_body:
                 # Sometimes structure differs, check basic 'products' key if 'body' is missing or flat
                 extracted_body = data

            products = extracted_body.get('products', [])
            
            if not products:
                 logger.warning(f"No products found in response for '{search_term}'")
            else:
                 logger.info(f"Successfully retrieved {len(products)} products.")
                 
            return products

        except requests.exceptions.RequestException as e:
            logger.error(f"Request Error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return []

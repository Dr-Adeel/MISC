from typing import List, Dict, Any
from datetime import datetime
import time
from market_provider import MarketProvider
from crawlbase_client import CrawlbaseClient

class AmazonProvider(MarketProvider):
    """
    Concrete implementation of MarketProvider for Amazon using Crawlbase.
    """

    def __init__(self, api_token: str):
        super().__init__("Amazon")
        self.client = CrawlbaseClient(api_token)

    def fetch_listings(self, product_name: str, domain: str, pages: List[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch Amazon listings for a given search term.
        Target specific pages.
        """
        all_products = []
        
        # Default to pages 1-3 if not specified
        if pages is None:
            pages = [1, 2, 3]
            
        # Loop through the specific list of pages
        for page in pages:
            products = self.client.get_products(domain, product_name, page=page)
            
            if products:
                # Add basic source info
                for p in products:
                    p['source_domain'] = domain
                
                all_products.extend(products)
            else:
                # If no products, stop this term
                break 
            
            # Respect rate limits
            time.sleep(2)
            
        return all_products

    def clean_data(self, raw_listings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich and standardize the product data.
        """
        cleaned_products = []
        
        for p in raw_listings:
            # Add collection timestamp
            p['collected_at'] = datetime.now().isoformat()
            
            # Ensure domain is set (already done in fetch but good to enforce)
            if 'source_domain' not in p:
                p['source_domain'] = 'unknown' # Should normally be passed or attached
            
            # Here you could add more cleaning logic (parsing prices, standardizing titles)
            # For now we preserve the raw structure as requested but ensure metadata
            
            cleaned_products.append(p)
            
        return cleaned_products

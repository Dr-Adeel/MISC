import json
import os
import time
from datetime import datetime
from config import CRAWLBASE_NORMAL_TOKEN, AMAZON_DOMAINS, SEARCH_TERMS, OUTPUT_FILE
from amazon_provider import AmazonProvider

def save_products(products, filename):
    """Save products to a JSON file with deduplication."""
    
    output_data = {
        "last_updated": datetime.now().isoformat(),
        "total_count": 0,
        "products": []
    }
    
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if "products" in existing_data:
                    output_data["products"] = existing_data["products"]
        except json.JSONDecodeError:
            pass
    
    # Deduplication using (ASIN, Domain)
    existing_keys = { 
        (p.get('asin'), p.get('source_domain')) 
        for p in output_data["products"] 
        if p.get('asin') 
    }
    
    new_count = 0
    for p in products:
        p['collected_at'] = datetime.now().isoformat()
        
        asin = p.get('asin')
        domain = p.get('source_domain')
        
        if asin and (asin, domain) not in existing_keys:
            output_data["products"].append(p)
            existing_keys.add((asin, domain))
            new_count += 1
            
    output_data["total_count"] = len(output_data["products"])
    output_data["last_updated"] = datetime.now().isoformat()
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {new_count} new products. Total: {output_data['total_count']}")

def main():
    if not CRAWLBASE_NORMAL_TOKEN:
        print("❌ Error: API Key not found. Please set CRAWLBASE_NORMAL_TOKEN in .env file.")
        return

    # Initialize Provider (Polymorphism / Abstraction)
    provider = AmazonProvider(CRAWLBASE_NORMAL_TOKEN)
    
    print(f"🚀 Starting Amazon Collection (Strategy: {provider.get_provider_name()}) across {len(AMAZON_DOMAINS)} domains...")
    
    for domain in AMAZON_DOMAINS:
        print(f"\n--- Processing Domain: {domain} ---")
        
        for term in SEARCH_TERMS:
            # 1. Fetch
            raw_products = provider.fetch_listings(term, domain)
            
            if raw_products:
                # 2. Clean
                cleaned_products = provider.clean_data(raw_products)
                
                # 3. Save (Persistence logic remains here or could be moved to a Repository)
                save_products(cleaned_products, OUTPUT_FILE)
            else:
                 print(f"No products found for '{term}' on {domain} (Pages 3-5)")

    print("\n✅ Collection Complete!")

if __name__ == "__main__":
    main()

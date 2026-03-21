import os
import json
from datetime import datetime
from config_extended import (
    CRAWLBASE_NORMAL_TOKEN, 
    AMAZON_DOMAINS, 
    SEARCH_TERMS, 
    OUTPUT_FILE_EXTENDED, 
    PREVIOUS_DATA_FILE
)
from amazon_provider import AmazonProvider

def load_existing_asins(filepath):
    """Load ASINs from a JSON file to create an exclusion set."""
    asins = set()
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for p in data.get("products", []):
                    asin = p.get("asin")
                    if asin:
                        asins.add(asin)
        except Exception as e:
            print(f"⚠️ Warning reading {filepath}: {e}")
    return asins

def save_products(products, filename, exclusion_set):
    """Save products if they are not in the exclusion set (previous data)."""
    
    output_data = {
        "last_updated": datetime.now().isoformat(),
        "total_count": 0,
        "products": []
    }
    
    # Load what we have accumulated in the NEW file so far
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if "products" in existing_data:
                    output_data["products"] = existing_data["products"]
        except:
            pass
    
    # Build set of what is ALREADY in the new file to avoid dupes there too
    current_file_asins = {p.get('asin') for p in output_data["products"] if p.get('asin')}
    
    new_count = 0
    for p in products:
        asin = p.get('asin')
        
        # CRITICAL CHECK: 
        # 1. Must have ASIN
        # 2. Must NOT be in the OLD file (exclusion_set)
        # 3. Must NOT be in the NEW file already (current_file_asins)
        
        if asin and asin not in exclusion_set and asin not in current_file_asins:
            p['collected_at'] = datetime.now().isoformat()
            p['source_domain'] = p.get('source_domain', 'amazon.com')
            
            output_data["products"].append(p)
            current_file_asins.add(asin) # Add to temp set so we don't add duplicate in same batch
            new_count += 1
            
    output_data["total_count"] = len(output_data["products"])
    output_data["last_updated"] = datetime.now().isoformat()
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {new_count} new products. Total: {output_data['total_count']}")

def main():
    if not CRAWLBASE_NORMAL_TOKEN:
        print("❌ Error: API Key not found.")
        return

    # 1. Load ASINs from the PREVIOUS dataset to ignore them
    print(f"Loading existing ASINs from {PREVIOUS_DATA_FILE} for deduplication...")
    old_asins = load_existing_asins(PREVIOUS_DATA_FILE)
    print(f"Found {len(old_asins)} existing products to ignore.")

    provider = AmazonProvider(CRAWLBASE_NORMAL_TOKEN)
    
    # Specific pages requested: 1, 2, 3, 5, 6
    PAGES_TO_FETCH = [1, 2,5, 6]
    
    print(f"🚀 Starting Extended Collection on {AMAZON_DOMAINS[0]} | Pages {PAGES_TO_FETCH}")
    
    for domain in AMAZON_DOMAINS:
        for term in SEARCH_TERMS:
            print(f"Searching: {term}...")
            try:
                # Fetch specific pages
                raw_products = provider.fetch_listings(term, domain, pages=PAGES_TO_FETCH)
                
                if raw_products:
                    cleaned_products = provider.clean_data(raw_products)
                    save_products(cleaned_products, OUTPUT_FILE_EXTENDED, old_asins)
                else:
                     print(f"No products found for '{term}'")
            except Exception as e:
                print(f"⚠️ Error: {e}")

    print("\n✅ Extended Collection with Deduplication Complete!")

if __name__ == "__main__":
    main()

"""
Comprehensive edge-case test suite for PricePulse pricing.
Tests every known failure pattern across ALL categories.

Failure patterns tested:
  1. Console/device ref matching accessories (PS5 controller, Xbox headset, etc.)
  2. Multi-language titles (French, Spanish, German, Arabic)
  3. Missing/budget brand detection
  4. Category routing collisions
  5. Accessory vs device misclassification
  6. Spec extraction false positives (numbers in titles)
  7. Ambiguous product names
  8. Pack/count/piece inflation bugs
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PRICEPULSE_SILENT'] = '1'

from predict_price import predict_price

# (title, category, expected_min, expected_max, description)
EDGE_CASES = [
    # ══════════════════════════════════════════════════════════════
    # 1. CONSOLE/DEVICE ACCESSORIES getting DEVICE prices
    # ══════════════════════════════════════════════════════════════
    # PS5 accessories
    ("DualSense Wireless Controller PS5", "Electronics - Gaming", 40, 100, "PS5 controller, NOT console"),
    ("Sony, Manette PlayStation 5 officielle DualSense, Sans fil, Bluetooth", "Electronics - Gaming", 40, 100, "FR: PS5 controller"),
    ("PS5 DualSense Charging Station", "Electronics - Gaming", 15, 50, "PS5 charging dock"),
    ("PS5 HD Camera 1080p", "Electronics - Gaming", 30, 80, "PS5 camera accessory"),
    ("PS5 Media Remote Control", "Electronics - Gaming", 15, 50, "PS5 remote"),
    ("Hogwarts Legacy PS5 Game", "Electronics - Gaming", 20, 70, "PS5 game, NOT console"),
    ("PS5 Console Cover Plate Cosmic Red", "Electronics - Gaming", 20, 70, "PS5 cover plate"),
    ("PS5 Pulse 3D Wireless Headset", "Electronics - Gaming", 50, 130, "PS5 headset"),

    # Xbox accessories
    ("Xbox Wireless Controller Carbon Black", "Electronics - Gaming", 35, 90, "Xbox controller"),
    ("Xbox Elite Wireless Controller Series 2", "Electronics - Gaming", 100, 220, "Xbox Elite controller"),
    ("Xbox Game Pass Ultimate 3 Month", "Electronics - Gaming", 20, 60, "Xbox subscription"),
    ("Xbox Series X Vertical Stand", "Electronics - Gaming", 10, 40, "Xbox stand"),
    ("Starfield Xbox Series X Game", "Electronics - Gaming", 20, 70, "Xbox game"),
    ("Xbox Wireless Headset", "Electronics - Gaming", 50, 130, "Xbox headset"),

    # Nintendo Switch accessories
    ("Nintendo Switch Pro Controller", "Electronics - Gaming", 40, 100, "Switch controller"),
    ("Joy-Con Pair Neon Red Blue", "Electronics - Gaming", 50, 110, "Joy-Con pair"),
    ("Nintendo Switch Carrying Case", "Electronics - Gaming", 10, 40, "Switch case"),
    ("Nintendo Switch Screen Protector 2 Pack", "Electronics - Gaming", 5, 20, "Switch screen protector"),
    ("Mario Kart 8 Deluxe Nintendo Switch", "Electronics - Gaming", 25, 65, "Switch game"),
    ("Nintendo Switch Dock Set", "Electronics - Gaming", 30, 80, "Switch dock"),

    # ══════════════════════════════════════════════════════════════
    # 2. PHONE ACCESSORIES getting PHONE prices
    # ══════════════════════════════════════════════════════════════
    ("iPhone 15 Pro Max Silicone Case MagSafe", "Electronics - Mobile & Accessories", 10, 60, "iPhone case"),
    ("Samsung Galaxy S24 Ultra Screen Protector 3 Pack", "Electronics - Mobile & Accessories", 5, 25, "Galaxy protector"),
    ("USB-C Lightning Cable for iPhone 2m", "Electronics - Mobile & Accessories", 5, 25, "iPhone cable"),
    ("Pixel 9 Pro Clear Case Slim", "Electronics - Mobile & Accessories", 8, 35, "Pixel case"),
    ("AirPods Pro 2 Replacement Ear Tips", "Electronics - Mobile & Accessories", 5, 20, "AirPods tips"),
    ("MagSafe Charger for iPhone", "Electronics - Mobile & Accessories", 15, 50, "MagSafe charger"),
    ("Galaxy S24 Ultra S Pen Replacement", "Electronics - Mobile & Accessories", 10, 40, "S-Pen replacement"),

    # ══════════════════════════════════════════════════════════════
    # 3. LAPTOP ACCESSORIES getting LAPTOP prices
    # ══════════════════════════════════════════════════════════════
    ("MacBook Pro 16 inch Laptop Sleeve", "Electronics - Computers", 15, 50, "MacBook sleeve"),
    ("MacBook Air M3 Charger 67W USB-C", "Electronics - Computers", 20, 60, "MacBook charger"),
    ("Dell XPS 15 Replacement Battery", "Electronics - Computers", 30, 80, "Dell battery"),
    ("ThinkPad USB-C Docking Station", "Electronics - Computers", 50, 200, "ThinkPad dock"),

    # ══════════════════════════════════════════════════════════════
    # 4. MULTI-LANGUAGE TITLES
    # ══════════════════════════════════════════════════════════════
    # French
    ("Apple iPhone 15 Pro Max 256 Go Titane Noir", "Electronics - Mobile & Accessories", 700, 1400, "FR: iPhone 15 Pro Max"),
    ("Samsung Galaxy S24 Ultra 512 Go Noir", "Electronics - Mobile & Accessories", 800, 1500, "FR: Galaxy S24 Ultra"),
    ("Casque Sony WH-1000XM5 Sans fil Réduction de bruit", "Electronics - Audio", 200, 420, "FR: Sony headphones"),
    ("Aspirateur Robot iRobot Roomba j9+ Auto-Vidage", "Electronics - Smart Home", 400, 900, "FR: Roomba j9"),
    ("Dyson V15 Detect Aspirateur Balai Sans Fil", "Home & Kitchen - Appliances", 350, 750, "FR: Dyson V15"),
    ("Machine à café Nespresso Vertuo Next", "Home & Kitchen - Appliances", 80, 250, "FR: Nespresso machine"),
    ("Écouteurs Apple AirPods Pro 2ème génération USB-C", "Electronics - Audio", 150, 320, "FR: AirPods Pro 2"),
    ("Téléviseur Samsung 65 pouces QLED 4K", "Electronics - TV & Video", 600, 1400, "FR: Samsung 65 TV"),
    ("Ordinateur Portable HP Pavilion 15 Core i5", "Electronics - Computers", 350, 800, "FR: HP Pavilion laptop"),

    # Spanish
    ("Auriculares Inalámbricos Sony WH-1000XM5", "Electronics - Audio", 200, 420, "ES: Sony headphones"),
    ("Aspiradora Robot Roomba j9+", "Electronics - Smart Home", 400, 900, "ES: Roomba j9"),
    ("Consola PlayStation 5 con Lector de Discos", "Electronics - Gaming", 350, 550, "ES: PS5 console"),
    ("Mando Inalámbrico DualSense PS5 Negro", "Electronics - Gaming", 40, 100, "ES: DualSense controller"),

    # German
    ("Sony WH-1000XM5 Kabellose Kopfhörer mit Noise Cancelling", "Electronics - Audio", 200, 420, "DE: Sony headphones"),
    ("Samsung Galaxy S24 Ultra 256GB Schwarz", "Electronics - Mobile & Accessories", 800, 1500, "DE: Galaxy S24 Ultra"),
    ("PlayStation 5 DualSense Wireless-Controller Schwarz", "Electronics - Gaming", 40, 100, "DE: PS5 controller"),

    # Arabic mixed
    ("ايفون iPhone 15 Pro Max 256GB", "Electronics - Mobile & Accessories", 700, 1400, "AR: iPhone 15 Pro Max"),

    # ══════════════════════════════════════════════════════════════
    # 5. BUDGET & UNKNOWN BRANDS
    # ══════════════════════════════════════════════════════════════
    ("infinix hot 30", "Electronics - Mobile & Accessories", 70, 200, "Budget phone: Infinix"),
    ("Tecno Spark 20 Pro 256GB", "Electronics - Mobile & Accessories", 90, 250, "Budget phone: Tecno"),
    ("itel A60s 32GB", "Electronics - Mobile & Accessories", 40, 130, "Ultra-budget: itel"),
    ("Realme GT 5 Pro 256GB 12GB RAM", "Electronics - Mobile & Accessories", 200, 500, "Mid-range: Realme"),
    ("Vivo V30 Pro 5G 256GB", "Electronics - Mobile & Accessories", 200, 450, "Mid-range: Vivo"),
    ("OUKITEL WP35 Rugged Smartphone 5G", "Electronics - Mobile & Accessories", 100, 400, "Unknown brand phone"),
    ("Blackview BV9300 Pro Rugged Phone 12GB RAM", "Electronics - Mobile & Accessories", 100, 450, "Unknown brand phone"),

    # Unknown/generic brands in other categories
    ("TOZO T6 True Wireless Earbuds", "Electronics - Audio", 15, 60, "Budget earbuds"),
    ("INIU Portable Charger 10000mAh Power Bank", "Electronics - Mobile & Accessories", 10, 40, "Budget power bank"),
    ("Anker Soundcore Life Q30 Headphones", "Electronics - Audio", 40, 100, "Budget headphones"),
    ("ELEGOO Saturn 3 Ultra Resin 3D Printer", "Electronics - Computers", 200, 700, "3D printer"),

    # ══════════════════════════════════════════════════════════════
    # 6. SPEC EXTRACTION FALSE POSITIVES
    # ══════════════════════════════════════════════════════════════
    ("Crayola 64 Count Crayons", "Toys & Games", 3, 15, "64 count = NOT 64GB"),
    ("Bounty Paper Towels 12 Count Mega Rolls", "Health & Household", 10, 40, "12 count = NOT 12GB"),
    ("Pampers Size 4 150 Count", "Baby Products", 20, 60, "150 count = NOT 150GB"),
    ("Duracell AA Batteries 48 Count", "Health & Household", 15, 40, "48 count ≠ storage"),
    ("Post-it Notes 3x3 inches 24 Pack", "Office Products", 10, 35, "24 pack = 24 pads"),
    ("Sharpie Permanent Markers 24 Count", "Office Products", 8, 30, "24 count markers"),
    ("K-Cups Coffee Pods 72 Count", "Health & Household", 20, 60, "72 count pods"),

    # ══════════════════════════════════════════════════════════════
    # 7. AMBIGUOUS PRODUCT NAMES
    # ══════════════════════════════════════════════════════════════
    ("Apple", "Books & Media", 3, 20, "Book called Apple, NOT Apple brand tech"),
    ("Ring", "Clothing & Accessories", 5, 200, "Jewelry ring, NOT Ring doorbell"),
    ("Echo Dot 5th Gen", "Electronics - Smart Home", 25, 65, "Echo Dot smart speaker"),
    ("Kindle Paperwhite 16GB 2024", "Electronics - Computers", 100, 200, "Kindle e-reader"),
    ("Fire TV Stick 4K Max", "Electronics - TV & Video", 30, 70, "Fire TV streaming"),
    ("Surface Pro 10 Core i7 16GB", "Electronics - Computers", 800, 1800, "Surface Pro tablet/laptop"),
    ("Galaxy Tab S9 Ultra 256GB", "Electronics - Computers", 700, 1400, "Samsung tablet"),

    # ══════════════════════════════════════════════════════════════
    # 8. KNOWN PREMIUM PRODUCTS (should NOT be cheap)
    # ══════════════════════════════════════════════════════════════
    ("Dyson Supersonic Hair Dryer", "Beauty & Personal Care", 300, 550, "Premium hair dryer"),
    ("KitchenAid Artisan Stand Mixer 5 Quart", "Home & Kitchen - Appliances", 250, 500, "Premium mixer"),
    ("Le Creuset Dutch Oven 7.25 Quart Round", "Home & Kitchen - Appliances", 280, 500, "Premium cookware"),
    ("Vitamix E310 Explorian Blender", "Home & Kitchen - Appliances", 250, 450, "Premium blender"),
    ("Weber Genesis Gas Grill 3 Burner", "Garden & Outdoor", 600, 1200, "Premium grill"),
    ("Sonos Arc Soundbar Dolby Atmos", "Electronics - Audio", 600, 1000, "Premium soundbar"),
    ("Canada Goose Chilliwack Bomber Jacket", "Clothing & Accessories", 500, 1400, "Premium jacket"),
    ("Herman Miller Aeron Chair Size B", "Home & Kitchen - Furniture", 800, 2000, "Premium office chair"),
    ("Rolex Submariner Date Watch", "Clothing & Accessories", 5000, 20000, "Luxury watch"),

    # ══════════════════════════════════════════════════════════════
    # 9. KNOWN CHEAP PRODUCTS (should NOT be expensive)
    # ══════════════════════════════════════════════════════════════
    ("BIC Cristal Ballpoint Pen 10 Pack", "Office Products", 2, 15, "Cheap pens"),
    ("Scotch Tape 6 Pack", "Office Products", 5, 20, "Cheap tape"),
    ("Ziploc Sandwich Bags 90 Count", "Health & Household", 3, 12, "Plastic bags"),
    ("Paper Mate Flair Felt Tip Pens 16 Pack", "Office Products", 8, 25, "Felt tip pens"),
    ("Elmer's Glue Stick 12 Pack", "Office Products", 3, 15, "Glue sticks"),
    ("Amazon Basics AA Batteries 20 Pack", "Health & Household", 8, 25, "Basic batteries"),
    ("Glad Trash Bags 13 Gallon 80 Count", "Health & Household", 8, 25, "Trash bags"),

    # ══════════════════════════════════════════════════════════════
    # 10. CAMERA/AUDIO/TV ACCESSORIES
    # ══════════════════════════════════════════════════════════════
    ("Canon EF 50mm f/1.8 STM Lens", "Electronics - Cameras", 80, 200, "Camera lens, not camera body"),
    ("Sony A7 IV Camera Battery NP-FZ100", "Electronics - Cameras", 15, 50, "Camera battery"),
    ("GoPro Hero 13 Protective Housing", "Electronics - Cameras", 15, 50, "GoPro case"),
    ("DJI Mini 4 Pro Propeller Guard", "Electronics - Cameras", 10, 40, "Drone propeller guard"),
    ("Samsung 65 inch TV Wall Mount Bracket", "Electronics - TV & Video", 15, 60, "TV mount, NOT TV"),
    ("LG OLED TV Remote Control Replacement", "Electronics - TV & Video", 10, 40, "TV remote"),
    ("Bose Soundbar Wall Mount Kit", "Electronics - Audio", 15, 50, "Soundbar mount"),
    ("AirPods Max Ear Cushions Replacement", "Electronics - Audio", 30, 80, "AirPods Max cushions"),

    # ══════════════════════════════════════════════════════════════
    # 11. SET / BUNDLE / MULTI-PACK
    # ══════════════════════════════════════════════════════════════
    ("Hanes Men's T-Shirt 6 Pack White", "Clothing & Accessories", 12, 35, "6-pack t-shirts"),
    ("Fruit of the Loom Boxer Briefs 7 Pack", "Clothing & Accessories", 12, 35, "7-pack underwear"),
    ("LEGO Star Wars Millennium Falcon 75375 Building Set 921 Pieces", "Toys & Games", 60, 140, "LEGO 921 pieces"),
    ("Crayola Markers 80 Colors Art Set", "Books & Media", 8, 30, "Art markers set"),
]

def run_tests():
    passed = 0
    failed = 0
    failures = []

    print(f"\n{'='*110}")
    print(f"  {'PRODUCT':<60} {'PRICE':>8}  {'EXPECTED':>14}  {'REF':<25} {'STATUS'}")
    print(f"{'='*110}")

    current_section = None
    for title, category, exp_min, exp_max, desc in EDGE_CASES:
        # Print section headers
        section = desc.split(":")[0] if ":" in desc else desc.split(",")[0]
        r = predict_price(title, category)
        price = r['predicted_price']
        ref = r.get('reference_used', '') or ''

        short_title = title[:58] + '..' if len(title) > 60 else title
        exp_str = f"${exp_min}-${exp_max}"

        if exp_min <= price <= exp_max:
            status = "✅"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
            failures.append((title, category, price, exp_min, exp_max, desc, ref))

        print(f"  {short_title:<60} ${price:>7.2f}  {exp_str:>14}  {ref:<25} {status}")

    print(f"\n{'='*110}")
    print(f"  RESULTS: {passed} passed, {failed} failed out of {passed+failed} tests")
    print(f"{'='*110}")

    if failures:
        print(f"\n  ❌ FAILURES:")
        print(f"  {'─'*106}")
        for title, cat, price, emin, emax, desc, ref in failures:
            direction = "TOO HIGH" if price > emax else "TOO LOW"
            print(f"  {desc}")
            print(f"    Title:    {title[:80]}")
            print(f"    Category: {cat}")
            print(f"    Got:      ${price:.2f} ({direction}) — Expected: ${emin}-${emax}")
            print(f"    Ref:      {ref}")
            print(f"  {'─'*106}")

    return failed

if __name__ == '__main__':
    failed = run_tests()
    sys.exit(1 if failed else 0)

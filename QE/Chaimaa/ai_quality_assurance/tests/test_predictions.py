"""Comprehensive prediction tests covering ALL 24 product categories."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PRICEPULSE_SILENT'] = '1'
from predict_price import predict_price

test_cases = [
    # ═══════════════════════════════════════════════════════════════
    # ELECTRONICS - COMPUTERS
    # ═══════════════════════════════════════════════════════════════
    ('acer core i7 8th gen', 'Electronics - Computers', 'Laptops'),
    ('acer core i7 8th gen 256GB storage 12 RAM', 'Electronics - Computers', 'Laptops'),
    ('Dell XPS 15 Core i9 32GB RAM 1TB SSD', 'Electronics - Computers', 'Laptops'),
    ('MacBook Air M3 256GB', 'Electronics - Computers', 'Laptops'),
    ('HP Pavilion 15 Core i5 12th gen 8GB RAM 512GB SSD', 'Electronics - Computers', 'Laptops'),
    ('Lenovo ThinkPad T14 Ryzen 7 16GB RAM', 'Electronics - Computers', 'Laptops'),
    ('ASUS ROG Strix Gaming Laptop RTX 4070 32GB RAM', 'Electronics - Computers', 'Laptops'),

    # ═══════════════════════════════════════════════════════════════
    # ELECTRONICS - MOBILE & ACCESSORIES
    # ═══════════════════════════════════════════════════════════════
    ('iPhone 15 Pro Max 256GB', 'Electronics - Mobile & Accessories', 'Phones'),
    ('Samsung Galaxy S24 Ultra 512GB', 'Electronics - Mobile & Accessories', 'Phones'),
    ('Google Pixel 9 Pro 128GB', 'Electronics - Mobile & Accessories', 'Phones'),
    ('USB Cable for iPhone 15', 'Electronics - Mobile & Accessories', 'Phones'),
    ('iPhone 15 Case Silicone MagSafe', 'Electronics - Mobile & Accessories', 'Phones'),
    ('Phone Tripod 86" Tall with Remote', 'Electronics - Mobile & Accessories', 'Phones'),

    # ═══════════════════════════════════════════════════════════════
    # ELECTRONICS - AUDIO
    # ═══════════════════════════════════════════════════════════════
    ('Sony WH-1000XM5 Headphones', 'Electronics - Audio', None),
    ('Apple AirPods Pro 2nd Generation USB-C', 'Electronics - Audio', None),
    ('JBL Flip 6 Portable Bluetooth Speaker Waterproof', 'Electronics - Audio', None),
    ('Bose QuietComfort Ultra Earbuds', 'Electronics - Audio', None),
    ('Shure SM7B Dynamic Microphone', 'Electronics - Audio', None),
    ('Sonos Arc Soundbar Dolby Atmos', 'Electronics - Audio', None),

    # ═══════════════════════════════════════════════════════════════
    # ELECTRONICS - TV & VIDEO
    # ═══════════════════════════════════════════════════════════════
    ('Samsung 65 inch QLED 4K Smart TV', 'Electronics - TV & Video', None),
    ('LG 55 inch OLED 4K Smart TV', 'Electronics - TV & Video', None),
    ('TCL 50 inch 4K UHD Roku Smart TV', 'Electronics - TV & Video', None),
    ('Sony 75 inch Bravia XR OLED', 'Electronics - TV & Video', None),

    # ═══════════════════════════════════════════════════════════════
    # ELECTRONICS - CAMERAS
    # ═══════════════════════════════════════════════════════════════
    ('Sony A7 IV Mirrorless Full Frame Camera', 'Electronics - Cameras', None),
    ('GoPro Hero 13 Black Action Camera 5K', 'Electronics - Cameras', None),
    ('DJI Mini 4 Pro Drone with Camera 4K', 'Electronics - Cameras', None),
    ('Logitech C920 HD Webcam 1080p', 'Electronics - Cameras', None),

    # ═══════════════════════════════════════════════════════════════
    # ELECTRONICS - GAMING
    # ═══════════════════════════════════════════════════════════════
    ('PlayStation 5 Console', 'Electronics - Gaming', None),
    ('Nintendo Switch OLED Model', 'Electronics - Gaming', None),
    ('Xbox Series X 1TB Console', 'Electronics - Gaming', None),
    ('Meta Quest 3 VR Headset 128GB', 'Electronics - Gaming', None),
    ('CATAN Board Game Strategy', 'Electronics - Gaming', None),
    ('DualSense Wireless Controller PS5', 'Electronics - Gaming', None),

    # ═══════════════════════════════════════════════════════════════
    # ELECTRONICS - WEARABLES
    # ═══════════════════════════════════════════════════════════════
    ('Apple Watch Series 10', 'Electronics - Wearables', None),
    ('Garmin Fenix 8 Sapphire Smartwatch', 'Electronics - Wearables', None),
    ('Fitbit Charge 6 Activity Tracker', 'Electronics - Wearables', None),
    ('Samsung Galaxy Watch 7 40mm', 'Electronics - Wearables', None),

    # ═══════════════════════════════════════════════════════════════
    # ELECTRONICS - SMART HOME
    # ═══════════════════════════════════════════════════════════════
    ('Ring Video Doorbell Pro 2', 'Electronics - Smart Home', None),
    ('Nest Learning Thermostat 4th Gen', 'Electronics - Smart Home', None),
    ('Roomba j9+ Robot Vacuum', 'Electronics - Smart Home', None),
    ('Philips Hue Smart Bulb Starter Kit', 'Electronics - Smart Home', None),
    ('Amazon Echo Show 10 Smart Display', 'Electronics - Smart Home', None),

    # ═══════════════════════════════════════════════════════════════
    # CLOTHING & ACCESSORIES
    # ═══════════════════════════════════════════════════════════════
    ('Nike Air Jordan 1 Mid Basketball Shoes', 'Clothing & Accessories', None),
    ('Adidas Ultra Boost 22 Running Shoes', 'Clothing & Accessories', None),
    ('North Face Thermoball Jacket', 'Clothing & Accessories', None),
    ("Levi's 501 Original Fit Jeans", 'Clothing & Accessories', None),
    ('New Balance 990v6 Made in USA', 'Clothing & Accessories', None),
    ('Canada Goose Expedition Parka', 'Clothing & Accessories', None),
    ('Ray-Ban Aviator Sunglasses', 'Clothing & Accessories', None),
    ('Hanes 6-Pack Cotton T-Shirts', 'Clothing & Accessories', None),

    # ═══════════════════════════════════════════════════════════════
    # HOME & KITCHEN - APPLIANCES
    # ═══════════════════════════════════════════════════════════════
    ('KitchenAid Artisan Stand Mixer', 'Home & Kitchen - Appliances', None),
    ('Dyson V15 Detect Cordless Vacuum', 'Home & Kitchen - Appliances', None),
    ('Ninja Foodi 12-in-1 Air Fryer Oven', 'Home & Kitchen - Appliances', None),
    ('Vitamix A3500 Professional Blender', 'Home & Kitchen - Appliances', None),
    ('Instant Pot Duo 7-in-1 6 Quart', 'Home & Kitchen - Appliances', None),
    ('Keurig K-Supreme Coffee Maker', 'Home & Kitchen - Appliances', None),
    ('Le Creuset Dutch Oven 5.5 Quart', 'Home & Kitchen - Appliances', None),

    # ═══════════════════════════════════════════════════════════════
    # HOME & KITCHEN - FURNITURE
    # ═══════════════════════════════════════════════════════════════
    ('Queen Memory Foam Mattress', 'Home & Kitchen - Furniture', None),
    ('Ergonomic Office Chair Lumbar Support', 'Home & Kitchen - Furniture', None),
    ('L-Shaped Desk with Drawers', 'Home & Kitchen - Furniture', None),
    ('King Size Bed Frame Platform Wood', 'Home & Kitchen - Furniture', None),
    ('Sectional Sofa with Chaise', 'Home & Kitchen - Furniture', None),
    ('Corner Bookshelf with LED Lights 6 Drawers', 'Home & Kitchen - Furniture', None),

    # ═══════════════════════════════════════════════════════════════
    # HOME & KITCHEN - DECOR
    # ═══════════════════════════════════════════════════════════════
    ('Washable Area Rug 8x10 Non-Slip', 'Home & Kitchen - Decor', None),
    ('Blackout Curtains 84 inch Set of 2', 'Home & Kitchen - Decor', None),
    ('Scented Candle Gift Set 4 Pack', 'Home & Kitchen - Decor', None),
    ('Table Lamp Modern Glass Shade', 'Home & Kitchen - Decor', None),
    ('Flower Vase 7.5 inch Glass', 'Home & Kitchen - Decor', None),

    # ═══════════════════════════════════════════════════════════════
    # TOOLS & HOME IMPROVEMENT
    # ═══════════════════════════════════════════════════════════════
    ('DeWalt 20V MAX Drill Combo Kit', 'Tools & Home Improvement', None),
    ('Milwaukee M18 FUEL Circular Saw', 'Tools & Home Improvement', None),
    ('Makita 18V LXT Impact Driver Kit', 'Tools & Home Improvement', None),
    ('Stanley FatMax Tape Measure 25ft', 'Tools & Home Improvement', None),
    ('Ryobi 40V Cordless Leaf Blower', 'Tools & Home Improvement', None),

    # ═══════════════════════════════════════════════════════════════
    # GARDEN & OUTDOOR
    # ═══════════════════════════════════════════════════════════════
    ('EGO Self Propelled Lawn Mower 56V', 'Garden & Outdoor', None),
    ('Weber Spirit Gas Grill 3 Burner', 'Garden & Outdoor', None),
    ('Traeger Ironwood 885 Pellet Grill', 'Garden & Outdoor', None),
    ('Solo Stove Bonfire Fire Pit', 'Garden & Outdoor', None),
    ('Patio Umbrella 9ft Tilt Crank', 'Garden & Outdoor', None),
    ('Garden Hose 100ft Expandable', 'Garden & Outdoor', None),

    # ═══════════════════════════════════════════════════════════════
    # SPORTS & OUTDOORS
    # ═══════════════════════════════════════════════════════════════
    ('Peloton Bike+ Exercise Bike', 'Sports & Outdoors', None),
    ('Bowflex SelectTech 552 Dumbbell Set', 'Sports & Outdoors', None),
    ('Manduka PRO Yoga Mat 6mm 71 inch', 'Sports & Outdoors', None),
    ('YETI Tundra 45 Hard Cooler', 'Sports & Outdoors', None),
    ('Osprey Atmos AG 65 Backpack', 'Sports & Outdoors', None),
    ('Theragun Elite Massage Gun', 'Sports & Outdoors', None),
    ('Wilson Evolution Indoor Basketball', 'Sports & Outdoors', None),
    ('Coleman Sundome 4-Person Tent', 'Sports & Outdoors', None),

    # ═══════════════════════════════════════════════════════════════
    # AUTOMOTIVE
    # ═══════════════════════════════════════════════════════════════
    ('Vantrue N4 Dual Dash Cam 4K', 'Automotive', None),
    ('NOCO Boost Plus Jump Starter 1000A', 'Automotive', None),
    ('WeatherTech FloorLiner Front Row', 'Automotive', None),
    ('Thule Roof Rack Cargo Box', 'Automotive', None),

    # ═══════════════════════════════════════════════════════════════
    # BABY PRODUCTS
    # ═══════════════════════════════════════════════════════════════
    ('Graco Modes Stroller', 'Baby Products', None),
    ('Britax Convertible Car Seat', 'Baby Products', None),
    ('UPPAbaby Vista V3 Stroller', 'Baby Products', None),
    ('SNOO Smart Bassinet', 'Baby Products', None),
    ('Pampers Swaddlers Size 3 168 Count', 'Baby Products', None),
    ('Baby Bjorn Carrier One', 'Baby Products', None),

    # ═══════════════════════════════════════════════════════════════
    # BEAUTY & PERSONAL CARE
    # ═══════════════════════════════════════════════════════════════
    ('Dyson Airwrap Multi-Styler', 'Beauty & Personal Care', None),
    ('Oral-B iO Series 9 Toothbrush', 'Beauty & Personal Care', None),
    ('CeraVe Moisturizing Cream 19oz', 'Beauty & Personal Care', None),
    ('The Ordinary Niacinamide 10% Serum', 'Beauty & Personal Care', None),
    ('Braun Series 9 Electric Shaver', 'Beauty & Personal Care', None),
    ('Olaplex No. 3 Hair Perfector', 'Beauty & Personal Care', None),

    # ═══════════════════════════════════════════════════════════════
    # MUSICAL INSTRUMENTS
    # ═══════════════════════════════════════════════════════════════
    ('Fender Stratocaster Electric Guitar', 'Musical Instruments', None),
    ('Yamaha P-45 Digital Piano 88 Key', 'Musical Instruments', None),
    ('Squier Classic Vibe Stratocaster', 'Musical Instruments', None),
    ('Focusrite Scarlett 2i2 Audio Interface', 'Musical Instruments', None),
    ('Boss DS-1 Distortion Pedal', 'Musical Instruments', None),

    # ═══════════════════════════════════════════════════════════════
    # TOYS & GAMES
    # ═══════════════════════════════════════════════════════════════
    ('LEGO Star Wars 921 Pieces Set', 'Toys & Games', None),
    ('Power Wheels Ride On 12V', 'Toys & Games', None),
    ('Magna-Tiles 100 Piece Magnetic Building Set', 'Toys & Games', None),
    ('Pokemon 12" Large Snorlax Plush', 'Toys & Games', None),
    ('NERF Elite 2.0 Blaster', 'Toys & Games', None),
    ('Barbie Dreamhouse 3-Story Dollhouse', 'Toys & Games', None),

    # ═══════════════════════════════════════════════════════════════
    # PET SUPPLIES
    # ═══════════════════════════════════════════════════════════════
    ('Cat Tree 67 inch Tower', 'Pet Supplies', None),
    ('Large Dog Bed Orthopedic Memory Foam', 'Pet Supplies', None),
    ('KONG Classic Dog Toy Large', 'Pet Supplies', None),
    ('Purina Pro Plan Dog Food 35 lb', 'Pet Supplies', None),
    ('PetSafe Easy Walk Dog Harness', 'Pet Supplies', None),

    # ═══════════════════════════════════════════════════════════════
    # OFFICE PRODUCTS
    # ═══════════════════════════════════════════════════════════════
    ('Brother Laser Printer Wireless', 'Office Products', None),
    ('Pilot G2 Gel Pens 12 Pack', 'Office Products', None),
    ('AmazonBasics Shredder 12 Sheet', 'Office Products', None),
    ('Moleskine Classic Notebook', 'Office Products', None),
    ('Fellowes Laminator 9 inch', 'Office Products', None),

    # ═══════════════════════════════════════════════════════════════
    # BOOKS & MEDIA
    # ═══════════════════════════════════════════════════════════════
    ('Atomic Habits Hardcover James Clear', 'Books & Media', None),
    ('2026 Weekly Planner Monthly Calendar', 'Books & Media', None),
    ('Crayola Markers 80 Colors Art Set', 'Books & Media', None),
    ('The Beatles Abbey Road Vinyl Record', 'Books & Media', None),

    # ═══════════════════════════════════════════════════════════════
    # HEALTH & HOUSEHOLD
    # ═══════════════════════════════════════════════════════════════
    ('Levoit Air Purifier Large Room', 'Health & Household', None),
    ('Garden of Life Multivitamin for Women', 'Health & Household', None),
    ('Optimum Nutrition Gold Standard Whey 5lb', 'Health & Household', None),
    ('Brita Water Filter Pitcher', 'Health & Household', None),
    ('Tide Pods Laundry Detergent 42 Count', 'Health & Household', None),
]

print(f"\n{'='*95}")
print(f"  {'TITLE':<55} {'PRICE':>10}  {'TYPE':<15} REF")
print(f"{'='*95}")

prev_cat = None
for title, cat, subcat in test_cases:
    if cat != prev_cat:
        print(f"  {'─'*90}")
        print(f"  📂 {cat}")
        print(f"  {'─'*90}")
        prev_cat = cat

    r = predict_price(title, cat, subcat)
    ref = r.get('reference_used') or ''
    ptype = ''
    if ref.startswith('heuristic:'):
        ptype = ref.replace('heuristic:', '')
        ref_display = ''
    else:
        ref_display = ref

    print(f"  {title[:55]:<55} ${r['predicted_price']:>8.2f}  {ptype:<15} {ref_display}")

print(f"{'='*95}")
print(f"\n  Total products tested: {len(test_cases)}")

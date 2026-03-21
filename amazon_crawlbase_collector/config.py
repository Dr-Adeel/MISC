import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
CRAWLBASE_NORMAL_TOKEN = os.getenv("CRAWLBASE_NORMAL_TOKEN")
CRAWLBASE_JS_TOKEN = os.getenv("CRAWLBASE_JS_TOKEN")

# Amazon Domains to scrape
AMAZON_DOMAINS = [
    "amazon.com",   # US
]

# Output Configuration
OUTPUT_FILE = "amazon_products_crawlbase.json"

# ==============================================================================
# DATASET STRATEGY: BROAD DIVERSITY (NON-ELECTRONICS FOCUS)
# ==============================================================================
SEARCH_TERMS = [
    # HOME & KITCHEN
    "Mattress", "Pillows", "Blanket", "Duvet", "Sheets", "Curtains", "Blinds",
    "Rug", "Doormat", "Mirror", "Wall Art", "Clock", "Vase", "Candles",
    "Sofa", "Chair", "Table", "Desk", "Bookshelf", "Cabinet", "Wardrobe",
    "Cookware", "Bakeware", "Dinnerware", "Glassware", "Cutlery",
    "Blender", "Toaster", "Coffee Maker", "Kettle", "Microwave", "Mixer",
    "Storage", "Organizer", "Hangers", "Laundry Basket", "Trash Can",
    "Vacuum Cleaner", "Iron", "Steamer", "Air Purifier", "Fan", "Heater",
    
    # GARDEN & OUTDOORS
    "Planters", "Seeds", "Fertilizer", "Garden Tools", "Hose", "Sprinkler",
    "Lawn Mower", "Leaf Blower", "Gloves", "Shovel", "Rake", "Pruner",
    "Patio Furniture", "Hammock", "Umbrella", "Gazebo", "Fire Pit",
    "Grill", "BBQ", "Charcoal", "Cooler", "Picnic Basket",
    "Tent", "Sleeping Bag", "Camping Chair", "Lantern", "Flashlight",
    
    # PET SUPPLIES
    "Dog Food", "Cat Food", "Treats", "Dog Bed", "Cat Tree", "Crate",
    "Leash", "Collar", "Harness", "Pet Toys", "Scratching Post",
    "Aquarium", "Fish Food", "Bird Cage", "Bird Food", "Hamster Cage",
    
    # FASHION & APPAREL
    "T-Shirt", "Shirt", "Jeans", "Pants", "Shorts", "Dress", "Skirt",
    "Jacket", "Coat", "Hoodie", "Sweater", "Suit", "Blazer",
    "Underwear", "Socks", "Bras", "Sleepwear", "Robes",
    "Shoes", "Sneakers", "Boots", "Sandals", "Slippers", "Heels",
    "Hat", "Cap", "Beanie", "Gloves", "Scarf", "Belt", "Wallet",
    "Backpack", "Handbag", "Purse", "Luggage", "Suitcase",
    "Jewelry", "Necklace", "Ring", "Earrings", "Bracelet", "Watch",
    
    # BEAUTY & PERSONAL CARE
    "Shampoo", "Conditioner", "Hair Mask", "Hair Oil", "Hairspray",
    "Body Wash", "Soap", "Lotion", "Moisturizer", "Scrub", "Deodorant",
    "Face Wash", "Toner", "Serum", "Sunscreen", "Face Mask",
    "Makeup", "Foundation", "Mascara", "Lipstick", "Eyeliner", "Eyeshadow",
    "Perfume", "Cologne", "Nail Polish", "Makeup Brushes",
    "Razor", "Shaving Cream", "Trimmer", "Hair Dryer", "Straightener",
    "Vitamins", "Supplements", "Protein Powder", "First Aid",
    
    # SPORTS & FITNESS
    "Dumbbells", "Kettlebell", "Yoga Mat", "Resistance Bands", "Foam Roller",
    "Treadmill", "Exercise Bike", "Elliptical", "Jump Rope",
    "Soccer Ball", "Basketball", "Football", "Volleyball", "Tennis Racket",
    "Golf Clubs", "Golf Balls", "Baseball Bat", "Hockey Stick",
    "Helmet", "Knee Pads", "Water Bottle", "Gym Bag",
    "Bicycle", "Skateboard", "Scooter", "Roller Skates",
    
    # AUTOMOTIVE
    "Car Mat", "Seat Cover", "Steering Wheel Cover", "Sun Shade",
    "Car Wash", "Car Wax", "Tire Shine", "Glass Cleaner",
    "Motor Oil", "Antifreeze", "Wiper Blades", "Car Battery", "Jump Starter",
    "Tire Inflator", "Jack", "Wrench", "Screwdriver", "Drill", "Tool Set",
    
    # TOYS & HOBBIES
    "Lego", "Action Figures", "Dolls", "Barbie", "Soft Toys", "Plush",
    "Board Games", "Card Games", "Puzzles", "Chess",
    "Remote Control Car", "Drone", "Train Set", "Building Blocks",
    "Art Set", "Paints", "Canvas", "Sketchbook", "Markers", "Crayons",
    "Guitar", "Piano", "Keyboard", "Ukulele", "Violin", "Drums",
    "Microphone", "Headphones", "Speaker",
    
    # BABY
    "Diapers", "Wipes", "Baby Powder", "Diaper Bag",
    "Stroller", "Car Seat", "Baby Carrier", "High Chair", "Walker",
    "Crib", "Bassinet", "Baby Monitor", "Humidifier",
    "Baby Clothes", "Onesies", "Bibs", "Swaddle",
    "Baby Bottles", "Pacifier", "Teether", "Baby Toys", "Rattle",
    
    # OFFICE & SCHOOL
    "Notebook", "Journal", "Planner", "Paper", "Sticky Notes",
    "Pen", "Pencil", "Highlighter", "Marker", "Eraser", "Sharpener",
    "Folder", "Binder", "File Organizer", "Stapler", "Scissors", "Tape",
    "Desk", "Office Chair", "Lamp", "Whiteboard", "Calculator", "Backpack"
]

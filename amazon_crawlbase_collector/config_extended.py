import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
CRAWLBASE_NORMAL_TOKEN = os.getenv("CRAWLBASE_NORMAL_TOKEN")

# Target Domain
AMAZON_DOMAINS = [
    "amazon.com"
]

# Output Configuration
OUTPUT_FILE_EXTENDED = "amazon_products_crawlbase_extended.json"
PREVIOUS_DATA_FILE = "amazon_products_crawlbase.json"

# ==============================================================================
# MOST SEARCHED AMAZON KEYWORDS (High Volume)
# Optimized for broad data collection
# ==============================================================================
SEARCH_TERMS = [
    # TECH & ELECTRONICS (High Volume)
    "Wireless Earbuds", "Bluetooth Headphones", "iPhone Case", "Lightning Cable", "USB-C Charger",
    "Power Bank", "Smart Watch", "Fitness Tracker", "Laptop Stand", "Mechanical Keyboard",
    "Gaming Mouse", "Webcam", "Ring Light", "Tripod", "Micro SD Card", "External Hard Drive",
    "HDMI Cable", "Ethernet Cable", "Surge Protector", "Extension Cord", "Batteries AA",
    "Batteries AAA", "Rechargeable Batteries", "Screen Protector", "Phone Mount for Car",
    
    # HOME & KITCHEN (Essentials)
    "Air Fryer", "Coffee Maker", "Blender", "Toaster", "Electric Kettle",
    "Rice Cooker", "Slow Cooker", "Food Scale", "Measuring Cups", "Tupperware Set",
    "Water Bottle", "Travel Mug", "Can Opener", "Kitchen Knife Set", "Cutting Board",
    "Dish Drying Rack", "Trash Can", "Laundry Hamper", "Hangers", "Shoe Rack",
    "Shower Curtain", "Bath Towels", "Bed Sheets Queen", "Pillow", "Mattress Topper",
    "Vacuum Cleaner", "Robot Vacuum", "Handheld Vacuum", "Air Purifier", "Humidifier",
    "Fan", "Space Heater", "Desk Lamp", "LED Strip Lights", "Curtains",
    
    # PERSONAL CARE & BEAUTY
    "Face Wash", "Moisturizer", "Vitamin C Serum", "Sunscreen", "Shampoo",
    "Conditioner", "Body Wash", "Deodorant", "Toothbrush", "Water Flosser",
    "Makeup Remover", "Cotton Pads", "Razor", "Shaving Cream", "Hair Dryer",
    "Hair Straightener", "Nail Clipper", "Tweezers", "Makeup Brushes", "Perfume",
    
    # FITNESS & OUTDOORS
    "Yoga Mat", "Resistance Bands", "Dumbbells", "Foam Roller", "Exercise Ball",
    "Running Shoes", "Gym Bag", "Camping Tent", "Sleeping Bag", "Flashlight",
    "Pocket Knife", "Cooler", "Grill Brush", "Gardening Gloves", "Hose Nozzle",
    
    # GAMING & TOYS
    "PS5 Controller", "Xbox Controller", "Nintendo Switch Case", "Gaming Headset",
    "Lego Set", "Board Games", "Card Games", "Puzzle", "Action Figure", "Doll",
    "Remote Control Car", "Drone", "Bubble Machine", "Plush Toy",
    
    # OFFICE & SCHOOL
    "Notebook", "Pens", "Pencils", "Highlighters", "Markers",
    "Scissors", "Stapler", "Tape", "Paper Clips", "Sticky Notes",
    "Desk Chair", "Mouse Pad", "Backpack", "Laptop Sleeve", "Calculator"
]

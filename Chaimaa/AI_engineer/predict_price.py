"""
🏪 PricePulse — Seller's Price Prediction Tool
================================================
ML-powered price prediction engine for Amazon products.
Combines GradientBoosting with reference price calibration
for accurate pricing recommendations.

Usage:
    from predict_price import predict_price, categories
    result = predict_price("iPhone 15 Pro 256GB", "Electronics")
"""

import sys
import joblib
import json
import numpy as np
import re
import os
from scipy.sparse import hstack, csr_matrix

# Fix Unicode output on Windows
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ================================================================
# LOAD MODEL & ARTIFACTS
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

_SILENT = os.environ.get('PRICEPULSE_SILENT', '') == '1'

if not _SILENT:
    print("\n" + "=" * 60)
    print("🏪 PricePulse — Price Prediction for Sellers")
    print("=" * 60)
    print("\n📦 Loading model...")

model = joblib.load(os.path.join(MODELS_DIR, 'model_price_predictor.pkl'))
tfidf = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
le_cat = joblib.load(os.path.join(MODELS_DIR, 'le_category.pkl'))
le_subcat = joblib.load(os.path.join(MODELS_DIR, 'le_subcategory.pkl'))
le_brand = joblib.load(os.path.join(MODELS_DIR, 'le_brand.pkl'))

with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'r', encoding='utf-8') as f:
    metadata = json.load(f)

known_brands = metadata['known_brands']
categories = metadata['categories']
subcategories = metadata['subcategories']

# Price medians/means for features (trained into the model)
cat_median_log_price = metadata.get('cat_median_log_price', {})
subcat_median_log_price = metadata.get('subcat_median_log_price', {})
brand_median_log_price = metadata.get('brand_median_log_price', {})
cat_mean_log_price = metadata.get('cat_mean_log_price', {})
brand_mean_log_price = metadata.get('brand_mean_log_price', {})
global_median_log_price = metadata.get('global_median_log_price', 3.5)
reference_prices = metadata.get('reference_prices', {})

if not _SILENT:
    print("✅ Model loaded successfully!")
    print(f"   Model R² = {metadata['r2_score']:.3f}")
    print(f"   MAE = ${metadata['mae']:.2f}")
    print(f"   📏 Reference prices: {len(reference_prices)} known products")


# ================================================================
# FEATURE EXTRACTION FUNCTIONS
# ================================================================

# Brand aliases: product names → brand
BRAND_ALIASES = {
    'iphone': 'apple', 'ipad': 'apple', 'macbook': 'apple', 'airpods': 'apple',
    'imac': 'apple', 'apple watch': 'apple', 'homepod': 'apple', 'mac mini': 'apple',
    'mac pro': 'apple', 'mac studio': 'apple',
    'galaxy': 'samsung', 'galaxy tab': 'samsung', 'galaxy watch': 'samsung',
    'galaxy buds': 'samsung', 'neo qled': 'samsung',
    'pixel': 'google', 'chromecast': 'google', 'nest': 'google',
    'surface': 'microsoft', 'xbox': 'microsoft',
    'playstation': 'sony', 'ps5': 'sony', 'bravia': 'sony',
    'wh-1000xm': 'sony', 'wf-1000xm': 'sony',
    'echo': 'amazon', 'kindle': 'amazon', 'fire tv': 'amazon', 'ring': 'amazon',
    'thinkpad': 'lenovo', 'ideapad': 'lenovo', 'yoga': 'lenovo',
    'xps': 'dell', 'inspiron': 'dell', 'alienware': 'dell',
    'pavilion': 'hp', 'envy': 'hp', 'spectre': 'hp', 'omen': 'hp',
    'rog': 'asus', 'zenbook': 'asus', 'vivobook': 'asus',
    'predator': 'acer', 'aspire': 'acer', 'nitro': 'acer', 'swift': 'acer',
    'quietcomfort': 'bose', 'soundlink': 'bose',
    'air jordan': 'nike', 'air max': 'nike', 'air force': 'nike',
    'ultraboost': 'adidas', 'ultra boost': 'adidas', 'yeezy': 'adidas',
    'roomba': 'irobot', 'braava': 'irobot',
    'kitchenaid artisan': 'kitchenaid',
    'instant pot': 'instant pot',
    'dyson v': 'dyson', 'dyson airwrap': 'dyson', 'dyson supersonic': 'dyson',
    'nintendo switch': 'nintendo', 'steam deck': 'valve',
    'gopro hero': 'gopro',
    'beats studio': 'beats', 'beats solo': 'beats', 'beats fit': 'beats',
    'fitbit sense': 'fitbit', 'fitbit versa': 'fitbit', 'fitbit charge': 'fitbit',
    # Budget / regional phone brands
    'infinix hot': 'infinix', 'infinix note': 'infinix', 'infinix zero': 'infinix',
    'infinix smart': 'infinix',
    'tecno spark': 'tecno', 'tecno camon': 'tecno', 'tecno pova': 'tecno',
    'tecno phantom': 'tecno',
    'redmi note': 'xiaomi', 'redmi 1': 'xiaomi', 'poco f': 'xiaomi',
    'poco x': 'xiaomi', 'poco m': 'xiaomi', 'poco c': 'xiaomi',
    'realme gt': 'realme', 'realme narzo': 'realme', 'realme c': 'realme',
    'honor magic': 'honor', 'honor x': 'honor', 'honor 2': 'honor',
    'honor 3': 'honor', 'honor 9': 'honor',
    'itel a': 'itel', 'itel p': 'itel', 'itel s': 'itel',
    'moto g': 'motorola', 'moto e': 'motorola', 'moto razr': 'motorola',
    'galaxy m': 'samsung', 'galaxy f': 'samsung',
    'nord ce': 'oneplus', 'nord n': 'oneplus',
}


def extract_brand(title):
    """Extract brand from product title using known brands + aliases."""
    t = title.lower()

    # 1. Direct brand match
    for brand in known_brands:
        if brand in t:
            return brand

    # 2. Alias match (product name → brand)
    # Sort by length descending for longest match first
    for alias, brand in sorted(BRAND_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if alias in t:
            return brand

    return 'unknown'


def extract_specs(title):
    """Extract technical specs from product title."""
    t = title.lower()
    specs = {}

    # Storage — distinguish storage vs RAM by context
    # "256 GB Storage" or "256GB SSD" → storage
    # "12 GB RAM" or "12GB DDR" → ram
    storage_matches = re.findall(r'(\d+)\s*gb\s*(?:storage|ssd|hdd|hard|drive|rom|emmc|flash|nvme|internal)', t)
    ram_matches = re.findall(r'(\d+)\s*gb\s*(?:ram|ddr|memory|lpddr)', t)

    if storage_matches:
        specs['storage_gb'] = int(storage_matches[0])
    elif ram_matches:
        # If we found RAM but no explicit storage, check for bare "XXX gb"
        all_gb = re.findall(r'(\d+)\s*gb', t)
        # Take the largest as storage (if >32) or first non-RAM match
        non_ram = [int(x) for x in all_gb if int(x) != int(ram_matches[0])]
        specs['storage_gb'] = max(non_ram) if non_ram else 0
    else:
        gb_match = re.search(r'(\d+)\s*gb', t)
        specs['storage_gb'] = int(gb_match.group(1)) if gb_match else 0

    # RAM
    if ram_matches:
        specs['ram_gb'] = int(ram_matches[0])
    else:
        # Try pattern: "12 RAM", "8gb ram"
        ram2 = re.search(r'(\d+)\s*(?:gb\s+)?ram', t)
        specs['ram_gb'] = int(ram2.group(1)) if ram2 else 0

    tb_match = re.search(r'(\d+)\s*tb', t)
    specs['storage_tb'] = int(tb_match.group(1)) if tb_match else 0

    inch_match = re.search(r'(\d+\.?\d*)\s*(?:inch|")', t)
    specs['screen_inch'] = float(inch_match.group(1)) if inch_match else 0

    mah_match = re.search(r'(\d+)\s*mah', t)
    specs['battery_mah'] = int(mah_match.group(1)) if mah_match else 0

    watt_match = re.search(r'(\d+)\s*(?:watt|w\b)', t)
    specs['wattage'] = int(watt_match.group(1)) if watt_match else 0

    pack_match = re.search(r'(?:pack of |(\d+)\s*-?\s*pack|set of (\d+))', t)
    if pack_match:
        nums = [g for g in pack_match.groups() if g]
        specs['pack_qty'] = int(nums[0]) if nums else 1
    else:
        specs['pack_qty'] = 1

    # Item count (NOT pack qty) — used for consumables; stored separately
    count_match = re.search(r'(\d+)\s*count', t)
    specs['item_count'] = int(count_match.group(1)) if count_match else 0

    # Piece count (building sets, puzzles)
    piece_match = re.search(r'(\d+)\s*pieces?', t)
    specs['piece_count'] = int(piece_match.group(1)) if piece_match else 0

    # CPU tier detection (not sent to model, used for heuristics)
    cpu_match = re.search(r'(?:core\s+)?(i[3579])(?:\s*[-]?\s*(\d+)\w*)?', t)
    if cpu_match:
        specs['_cpu_tier'] = cpu_match.group(1)  # i3, i5, i7, i9
    elif re.search(r'ryzen\s*(\d)', t):
        m = re.search(r'ryzen\s*(\d)', t)
        specs['_cpu_tier'] = f"ryzen_{m.group(1)}"
    elif re.search(r'\bm[1-4]\b', t):
        m = re.search(r'\b(m[1-4])\b', t)
        specs['_cpu_tier'] = m.group(1)
    else:
        specs['_cpu_tier'] = None

    # Generation detection
    gen_match = re.search(r'(\d{1,2})(?:th|st|nd|rd)\s*gen', t)
    specs['_generation'] = int(gen_match.group(1)) if gen_match else None

    return specs


def extract_quality_flags(title):
    """Extract quality/attribute flags from title."""
    t = title.lower()
    flags = {}
    flags['is_premium'] = int(bool(re.search(
        r'\b(premium|pro|professional|ultra|luxury|platinum|titanium|elite|advanced|deluxe)\b', t)))
    flags['is_budget'] = int(bool(re.search(
        r'\b(basic|mini|lite|cheap|budget|value|economy|starter|compact|portable)\b', t)))
    flags['is_wireless'] = int(bool(re.search(
        r'\b(wireless|bluetooth|wifi|wi-fi)\b', t)))
    flags['is_waterproof'] = int(bool(re.search(
        r'\b(waterproof|water.?resistant|ip\d+)\b', t)))
    flags['is_organic'] = int(bool(re.search(
        r'\b(organic|natural|eco|sustainable|biodegradable)\b', t)))
    flags['has_led'] = int(bool(re.search(
        r'\b(led|oled|amoled|lcd)\b', t)))
    flags['is_smart'] = int(bool(re.search(
        r'\b(smart|ai|alexa|google assistant|siri)\b', t)))
    return flags


def safe_label_encode(encoder, value):
    """Safely encode a label, using 0 for unseen labels."""
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return 0


# ================================================================
# REFERENCE PRICE MATCHING
# ================================================================
def find_reference_price(title):
    """
    Match product title against known reference prices.
    Uses longest-match-first strategy for accuracy.

    Strategy:
      1. Exact substring match (highest priority).
      2. All-words-present match: every word in the reference key
         appears somewhere in the title (handles "SNOO Smart Bassinet"
         matching key "snoo bassinet").

    Returns:
        (reference_price, matched_key) or (None, None)
    """
    t_lower = title.lower()

    # Sort by key length descending so "iphone 15 pro max" matches before "iphone 15"
    sorted_refs = sorted(reference_prices.items(), key=lambda x: len(x[0]), reverse=True)

    # Pass 1: exact substring
    for key, price in sorted_refs:
        if key in t_lower:
            return price, key

    # Pass 2: all words of key present in title (order-independent)
    t_words = set(t_lower.split())
    for key, price in sorted_refs:
        key_words = key.split()
        if len(key_words) >= 2 and all(w in t_words for w in key_words):
            return price, key

    return None, None


def _find_reference_excluding(title, exclude_keys):
    """
    Find a reference price but skip any ref key that contains
    one of the exclude_keys strings.

    Used after the console-accessory guard strips a console reference
    to find a more specific match (e.g. "dualsense" instead of "playstation 5").
    """
    t_lower = title.lower()
    sorted_refs = sorted(reference_prices.items(), key=lambda x: len(x[0]), reverse=True)

    for key, price in sorted_refs:
        if any(ex in key for ex in exclude_keys):
            continue  # skip console/device refs
        if key in t_lower:
            return price, key

    t_words = set(t_lower.split())
    for key, price in sorted_refs:
        if any(ex in key for ex in exclude_keys):
            continue
        key_words = key.split()
        if len(key_words) >= 2 and all(w in t_words for w in key_words):
            return price, key

    return None, None


def _apply_spec_adjustments(base_price, specs, flags):
    """
    Apply price adjustments based on detected specs.
    Larger storage, bigger screens, etc. justify higher prices.
    """
    adjustment = 1.0

    # Storage adjustments (relative to base configs)
    storage_gb = specs.get('storage_gb', 0)
    if storage_gb >= 1024:
        adjustment *= 1.25
    elif storage_gb >= 512:
        adjustment *= 1.12
    elif storage_gb >= 256:
        adjustment *= 1.0   # base
    elif storage_gb >= 128:
        adjustment *= 0.92

    storage_tb = specs.get('storage_tb', 0)
    if storage_tb >= 4:
        adjustment *= 1.30
    elif storage_tb >= 2:
        adjustment *= 1.15

    # Screen size
    screen = specs.get('screen_inch', 0)
    if screen >= 65:
        adjustment *= 1.20
    elif screen >= 50:
        adjustment *= 1.10

    # Pack quantity
    pack = specs.get('pack_qty', 1)
    if pack > 1:
        # Don't inflate price when it's a cheap consumable (ref < $30)
        if base_price < 30:
            adjustment *= min(pack * 0.20, 2.0)  # modest bump for cheap items
        else:
            adjustment *= min(pack * 0.85, 5.0)  # bulk discount per unit

    # Premium flag bump
    if flags.get('is_premium', 0):
        adjustment *= 1.08
    if flags.get('is_budget', 0):
        adjustment *= 0.90

    return base_price * adjustment


# ================================================================
# SMART HEURISTIC ESTIMATION (for ALL product categories)
# ================================================================

# ──── ELECTRONICS - COMPUTERS (Laptops/Desktops) ────
_CPU_BASE_PRICES = {
    'i9': 1300, 'i7': 800, 'i5': 550, 'i3': 380,
    'ryzen_9': 1200, 'ryzen_7': 750, 'ryzen_5': 500, 'ryzen_3': 380,
    'm4': 1300, 'm3': 1100, 'm2': 1000, 'm1': 900,
}
_LAPTOP_BRAND_FACTORS = {
    'apple': 1.30, 'microsoft': 1.15, 'dell': 1.05, 'hp': 1.00,
    'lenovo': 1.05, 'asus': 1.00, 'acer': 0.90, 'msi': 1.10,
    'razer': 1.25, 'samsung': 1.05, 'lg': 1.00,
}

# ──── ELECTRONICS - MOBILE (Phones) ────
_PHONE_BRAND_PRICES = {
    'apple': 699, 'samsung': 499, 'google': 499, 'sony': 599,
    'oneplus': 399, 'xiaomi': 299, 'motorola': 249, 'nokia': 199,
    'huawei': 399, 'oppo': 349, 'nothing': 349, 'zte': 199,
    'infinix': 149, 'tecno': 149, 'realme': 249, 'vivo': 299,
    'honor': 299, 'itel': 99, 'lava': 99, 'micromax': 99,
    'tcl': 199, 'alcatel': 149, 'blu': 129, 'umidigi': 149,
}

# ──── ELECTRONICS - TV ────
_TV_SIZE_PRICES = {
    (85, 999): 1600, (75, 84): 1200, (65, 74): 800, (55, 64): 500,
    (50, 54): 400, (43, 49): 300, (32, 42): 200, (0, 31): 150,
}

# ──── ELECTRONICS - AUDIO ────
_AUDIO_TYPE_PRICES = {
    'headphones_over': 150, 'headphones_on': 100, 'earbuds': 80,
    'speaker_portable': 60, 'speaker_home': 200, 'soundbar': 250,
    'microphone': 120, 'turntable': 200, 'receiver': 350,
    'studio_monitor': 250, 'amplifier': 300, 'karaoke': 40,
}
_AUDIO_BRAND_FACTORS = {
    'bose': 1.40, 'sony': 1.25, 'sennheiser': 1.35, 'bang & olufsen': 2.0,
    'apple': 1.30, 'beats': 1.10, 'jbl': 1.00, 'audio-technica': 1.15,
    'shure': 1.30, 'harman kardon': 1.20, 'sonos': 1.40, 'marshall': 1.15,
    'skullcandy': 0.70, 'anker': 0.75, 'samsung': 1.05,
}

# ──── ELECTRONICS - CAMERAS ────
_CAMERA_TYPE_PRICES = {
    'dslr': 800, 'mirrorless': 1000, 'action_cam': 300, 'webcam': 60,
    'security_cam': 50, 'drone': 400, 'dash_cam': 120, 'instant': 80,
    'point_shoot': 300, 'film': 150, 'trail_cam': 100,
}
_CAMERA_BRAND_FACTORS = {
    'canon': 1.10, 'nikon': 1.10, 'sony': 1.25, 'fujifilm': 1.15,
    'panasonic': 1.00, 'olympus': 0.95, 'leica': 2.50, 'gopro': 1.10,
    'dji': 1.20, 'insta360': 1.00,
}

# ──── ELECTRONICS - GAMING ────
_GAMING_TYPE_PRICES = {
    'console': 400, 'controller': 55, 'headset': 60, 'keyboard': 80,
    'mouse': 50, 'chair': 250, 'monitor': 300, 'capture_card': 150,
    'vr_headset': 350, 'steering_wheel': 250, 'board_game': 30,
    'card_game': 20, 'video_game': 50, 'memory_card': 25,
    'stand': 20, 'dock': 50,
}

# ──── ELECTRONICS - WEARABLES ────
_WEARABLE_BRAND_PRICES = {
    'apple': 350, 'samsung': 250, 'garmin': 300, 'fitbit': 150,
    'whoop': 200, 'amazfit': 80, 'suunto': 300, 'polar': 250,
    'withings': 180, 'fossil': 150, 'coros': 250,
}

# ──── ELECTRONICS - SMART HOME ────
_SMARTHOME_TYPE_PRICES = {
    'thermostat': 150, 'doorbell': 100, 'security_cam': 60,
    'speaker': 80, 'display': 130, 'plug': 20, 'switch': 30,
    'bulb': 15, 'lock': 200, 'hub': 100, 'vacuum_robot': 350,
    'air_purifier': 200, 'humidifier': 60, 'heater': 80,
}

# ──── CLOTHING & ACCESSORIES ────
_CLOTHING_TYPE_PRICES = {
    'shoes_athletic': 100, 'shoes_casual': 70, 'shoes_boots': 120,
    'shoes_dress': 90, 'shoes_sandals': 45, 'jacket': 100, 'coat': 150,
    'jeans': 45, 'pants': 40, 'dress': 50, 'suit': 200,
    'shirt': 30, 'tshirt': 20, 'hoodie': 45, 'sweater': 50,
    'shorts': 25, 'skirt': 35, 'underwear': 15, 'socks': 12,
    'hat': 25, 'watch': 150, 'sunglasses': 80, 'handbag': 120,
    'backpack': 60, 'belt': 30, 'scarf': 30, 'gloves': 25,
    'swimwear': 35, 'activewear': 45, 'pajamas': 30,
}
_CLOTHING_BRAND_FACTORS = {
    'nike': 1.25, 'adidas': 1.15, 'new balance': 1.15, 'under armour': 1.10,
    'north face': 1.40, 'patagonia': 1.50, 'columbia': 1.10,
    'canada goose': 3.00, 'moncler': 3.50, 'ralph lauren': 1.40,
    'levi': 1.00, 'calvin klein': 1.15, 'tommy hilfiger': 1.10,
    'coach': 1.80, 'michael kors': 1.60, 'gucci': 4.00, 'prada': 4.00,
    'burberry': 3.00, 'timberland': 1.20, 'dr martens': 1.30,
    'birkenstock': 1.15, 'skechers': 0.80, 'crocs': 0.80,
    'puma': 1.00, 'reebok': 0.90, 'converse': 0.85, 'vans': 0.85,
    'carhartt': 1.20, 'hanes': 0.50, 'fruit of the loom': 0.40,
}

# ──── HOME & KITCHEN - APPLIANCES ────
_APPLIANCE_TYPE_PRICES = {
    'blender': 80, 'mixer': 200, 'food_processor': 120, 'toaster': 40,
    'coffee_maker': 100, 'espresso': 350, 'microwave': 120,
    'air_fryer': 100, 'slow_cooker': 60, 'pressure_cooker': 80,
    'rice_cooker': 60, 'juicer': 80, 'ice_maker': 150,
    'dishwasher': 500, 'refrigerator': 800, 'washer': 600,
    'dryer': 600, 'oven': 500, 'range': 700, 'vacuum': 250,
    'iron': 30, 'sewing_machine': 150, 'dehydrator': 60,
    'waffle_maker': 30, 'griddle': 40, 'kettle': 40,
    'pan_set': 100, 'pot_set': 100, 'knife_set': 80, 'bakeware': 30,
    'cookware_set': 200, 'cutting_board': 25,
}
_APPLIANCE_BRAND_FACTORS = {
    'dyson': 1.80, 'kitchenaid': 1.50, 'vitamix': 1.60, 'breville': 1.40,
    'cuisinart': 1.15, 'ninja': 1.00, 'instant pot': 1.00,
    'keurig': 1.10, 'nespresso': 1.30, 'de\'longhi': 1.25,
    'shark': 1.00, 'bissell': 0.85, 'irobot': 1.30,
    'samsung': 1.10, 'lg': 1.05, 'whirlpool': 1.00, 'ge': 1.00,
    'bosch': 1.20, 'miele': 1.80, 'viking': 2.00,
    'all-clad': 1.60, 'le creuset': 1.80, 'staub': 1.60,
    'calphalon': 1.20, 'lodge': 0.80, 'hamilton beach': 0.60,
}

# ──── HOME & KITCHEN - FURNITURE ────
_FURNITURE_TYPE_PRICES = {
    'mattress': 400, 'mattress_topper': 80, 'bed_frame': 250,
    'sofa': 600, 'sectional': 1200, 'loveseat': 400,
    'desk': 200, 'office_chair': 200, 'dining_table': 350,
    'dining_chair': 100, 'bookshelf': 120, 'dresser': 250,
    'nightstand': 80, 'tv_stand': 120, 'coffee_table': 150,
    'end_table': 60, 'wardrobe': 300, 'cabinet': 200,
    'shoe_rack': 40, 'storage_ottoman': 60, 'bean_bag': 50,
    'bar_stool': 80, 'recliner': 400, 'futon': 250,
    'bunk_bed': 350, 'crib': 200, 'changing_table': 150,
    'vanity': 200, 'mirror': 60, 'rug': 80, 'curtain': 30,
    'pillow': 25, 'bedding_set': 60, 'comforter': 50,
}
_FURNITURE_BRAND_FACTORS = {
    'herman miller': 3.00, 'steelcase': 2.50, 'secretlab': 1.50,
    'ikea': 0.70, 'wayfair': 0.90, 'ashley': 1.00,
    'restoration hardware': 2.50, 'pottery barn': 1.80,
    'west elm': 1.50, 'crate & barrel': 1.60,
    'casper': 1.20, 'purple': 1.30, 'tempur-pedic': 1.80,
    'nectar': 0.90, 'tuft & needle': 0.85, 'serta': 1.00,
}

# ──── HOME & KITCHEN - DECOR ────
_DECOR_TYPE_PRICES = {
    'rug': 80, 'curtain': 30, 'pillow': 20, 'throw_blanket': 35,
    'wall_art': 40, 'picture_frame': 20, 'candle': 20,
    'lamp': 50, 'chandelier': 200, 'vase': 25, 'clock': 30,
    'mirror': 50, 'plant_pot': 15, 'tapestry': 25,
    'shelf_decor': 20, 'basket': 25, 'towel_set': 25,
    'shower_curtain': 25, 'bath_mat': 20, 'organizer': 25,
}

# ──── TOOLS & HOME IMPROVEMENT ────
_TOOL_TYPE_PRICES = {
    'drill': 100, 'saw': 150, 'sander': 60, 'grinder': 50,
    'impact_driver': 120, 'wrench_set': 30, 'screwdriver_set': 20,
    'pliers_set': 25, 'hammer': 20, 'tape_measure': 15,
    'level': 25, 'stud_finder': 30, 'multimeter': 40,
    'power_washer': 250, 'generator': 600, 'air_compressor': 200,
    'welding': 300, 'nail_gun': 200, 'router': 150,
    'workbench': 200, 'toolbox': 50, 'ladder': 150,
    'paint_sprayer': 120, 'heat_gun': 30, 'oscillating_tool': 80,
}
_TOOL_BRAND_FACTORS = {
    'dewalt': 1.20, 'milwaukee': 1.25, 'makita': 1.20,
    'bosch': 1.15, 'ryobi': 0.80, 'craftsman': 0.90,
    'ridgid': 1.00, 'hilti': 1.60, 'festool': 2.00,
    'snap-on': 2.00, 'stanley': 0.85, 'black+decker': 0.70,
    'kobalt': 0.80, 'ego': 1.15, 'greenworks': 0.90,
    'husqvarna': 1.30, 'stihl': 1.30,
}

# ──── GARDEN & OUTDOOR ────
_GARDEN_TYPE_PRICES = {
    'lawn_mower': 350, 'lawn_mower_riding': 2000, 'trimmer': 80,
    'leaf_blower': 100, 'chainsaw': 200, 'hedge_trimmer': 80,
    'pressure_washer': 250, 'grill': 400, 'smoker': 350,
    'patio_furniture': 500, 'patio_umbrella': 100, 'fire_pit': 200,
    'outdoor_light': 30, 'garden_hose': 30, 'sprinkler': 25,
    'shed': 600, 'greenhouse': 300, 'planter': 30, 'soil': 15,
    'seeds': 5, 'fertilizer': 20, 'garden_tools': 25,
    'wheelbarrow': 80, 'snow_blower': 500, 'pool': 300,
    'hot_tub': 3000, 'trampoline': 300, 'swing_set': 400,
    'bird_feeder': 25, 'bird_house': 20,
}

# ──── SPORTS & OUTDOORS ────
_SPORTS_TYPE_PRICES = {
    'treadmill': 800, 'exercise_bike': 500, 'elliptical': 600,
    'rowing_machine': 500, 'weight_bench': 200, 'dumbbell': 80,
    'dumbbell_set': 300, 'kettlebell': 40, 'barbell': 150,
    'resistance_band': 20, 'yoga_mat': 30, 'pull_up_bar': 30,
    'jump_rope': 15, 'punching_bag': 100, 'boxing_gloves': 40,
    'bicycle': 500, 'ebike': 1200, 'helmet': 50, 'ski': 400,
    'snowboard': 350, 'surfboard': 400, 'kayak': 500,
    'tent': 150, 'sleeping_bag': 60, 'hiking_boots': 120,
    'backpack_hiking': 150, 'fishing_rod': 60, 'fishing_reel': 80,
    'golf_club': 200, 'golf_set': 500, 'basketball': 30,
    'football': 25, 'soccer_ball': 25, 'baseball_bat': 50,
    'tennis_racket': 80, 'scooter': 150, 'skateboard': 60,
    'massage_gun': 100, 'foam_roller': 25, 'binoculars': 100,
    'flashlight': 25, 'cooler': 40, 'water_bottle': 25,
}
_SPORTS_BRAND_FACTORS = {
    'peloton': 2.00, 'bowflex': 1.30, 'nordictrack': 1.30,
    'rogue': 1.50, 'yeti': 1.60, 'osprey': 1.30,
    'north face': 1.30, 'patagonia': 1.40, 'rei': 1.10,
    'coleman': 0.80, 'ozark trail': 0.50, 'wilson': 1.00,
    'callaway': 1.20, 'titleist': 1.30, 'trek': 1.30,
    'specialized': 1.40, 'giant': 1.00, 'manduka': 1.40,
    'lululemon': 1.50, 'garmin': 1.30, 'theragun': 1.30,
}

# ──── AUTOMOTIVE ────
_AUTO_TYPE_PRICES = {
    'dash_cam': 120, 'jump_starter': 80, 'tire': 100,
    'tire_set': 400, 'car_cover': 60, 'seat_cover': 50,
    'floor_mat': 40, 'wiper': 15, 'headlight': 40,
    'car_battery': 120, 'oil': 25, 'air_filter': 15,
    'brake_pad': 40, 'car_charger': 15, 'phone_mount': 15,
    'car_stereo': 150, 'subwoofer': 100, 'amplifier': 150,
    'roof_rack': 200, 'hitch': 150, 'winch': 250,
    'jack': 40, 'tool_kit': 50, 'air_pump': 40,
    'car_wash': 15, 'wax': 15, 'car_vacuum': 40,
    'steam_cleaner': 60, 'obd_scanner': 30,
}

# ──── BABY PRODUCTS ────
_BABY_TYPE_PRICES = {
    'stroller': 250, 'car_seat': 200, 'high_chair': 120,
    'crib': 200, 'bassinet': 150, 'playpen': 100,
    'baby_monitor': 100, 'baby_swing': 120, 'bouncer': 60,
    'walker': 50, 'carrier': 80, 'diaper_bag': 40,
    'bottle': 15, 'breast_pump': 150, 'formula': 30,
    'diaper': 25, 'wipes': 15, 'pacifier': 8,
    'teether': 8, 'bib': 10, 'swaddle': 20,
    'baby_clothes': 15, 'baby_shoes': 15, 'baby_blanket': 20,
    'baby_bath': 25, 'baby_gate': 40, 'baby_toy': 15,
    'changing_table': 150, 'nursery_decor': 30,
}
_BABY_BRAND_FACTORS = {
    'uppababy': 2.00, 'bugaboo': 2.20, 'nuna': 1.80,
    'baby bjorn': 1.50, 'graco': 1.00, 'chicco': 1.05,
    'britax': 1.15, 'baby trend': 0.80, 'evenflo': 0.80,
    'fisher-price': 0.85, 'halo': 1.30, 'snoo': 3.00,
    'medela': 1.20, 'spectra': 1.10,
}

# ──── BEAUTY & PERSONAL CARE ────
_BEAUTY_TYPE_PRICES = {
    'hair_dryer': 60, 'hair_styler': 100, 'straightener': 50,
    'curling_iron': 40, 'hair_clipper': 40, 'beard_trimmer': 30,
    'electric_shaver': 60, 'electric_toothbrush': 80,
    'water_flosser': 50, 'face_wash': 15, 'moisturizer': 20,
    'serum': 25, 'sunscreen': 12, 'foundation': 25,
    'mascara': 12, 'lipstick': 15, 'eyeshadow': 20,
    'perfume': 60, 'cologne': 60, 'deodorant': 8,
    'shampoo': 12, 'conditioner': 12, 'hair_mask': 15,
    'body_wash': 10, 'body_lotion': 12, 'face_mask': 15,
    'nail_polish': 8, 'makeup_brush': 20, 'mirror_vanity': 30,
    'scale': 30, 'massage_device': 50, 'led_mask': 100,
}
_BEAUTY_BRAND_FACTORS = {
    'dyson': 3.00, 'foreo': 1.80, 'nuface': 2.00,
    'oral-b': 1.20, 'philips': 1.10, 'braun': 1.15,
    'panasonic': 1.05, 'revlon': 0.70, 'conair': 0.60,
    'olaplex': 1.40, 'kerastase': 1.60, 'moroccanoil': 1.40,
    'la mer': 3.00, 'estee lauder': 1.50, 'clinique': 1.30,
    'cerave': 0.80, 'neutrogena': 0.70, 'the ordinary': 0.60,
    'tatcha': 1.60, 'drunk elephant': 1.50, 'skinceuticals': 2.00,
}

# ──── MUSICAL INSTRUMENTS ────
_MUSIC_TYPE_PRICES = {
    'acoustic_guitar': 250, 'electric_guitar': 500, 'bass_guitar': 400,
    'ukulele': 50, 'violin': 200, 'cello': 800, 'flute': 200,
    'trumpet': 300, 'saxophone': 500, 'clarinet': 300,
    'drum_kit': 500, 'drum_pad': 100, 'cymbal': 100,
    'keyboard': 300, 'digital_piano': 500, 'synthesizer': 500,
    'midi_controller': 100, 'audio_interface': 150,
    'guitar_amp': 200, 'bass_amp': 200, 'pedal': 80,
    'microphone_music': 100, 'guitar_strings': 8,
    'drum_sticks': 8, 'tuner': 15, 'capo': 10,
    'music_stand': 20, 'case': 50, 'strap': 15,
}
_MUSIC_BRAND_FACTORS = {
    'fender': 1.30, 'gibson': 2.00, 'martin': 2.00, 'taylor': 1.80,
    'yamaha': 1.00, 'roland': 1.20, 'korg': 1.10, 'casio': 0.70,
    'ibanez': 0.90, 'epiphone': 0.70, 'squier': 0.55,
    'prs': 1.60, 'rickenbacker': 1.80, 'gretsch': 1.30,
    'pearl': 1.20, 'zildjian': 1.30, 'sabian': 1.10,
    'shure': 1.30, 'focusrite': 1.10, 'native instruments': 1.30,
    'boss': 1.10, 'line 6': 1.00, 'orange': 1.30,
    'marshall': 1.20, 'vox': 1.00,
}

# ──── TOYS & GAMES ────
_TOY_TYPE_PRICES = {
    'building_set': 40, 'action_figure': 20, 'doll': 25,
    'dollhouse': 80, 'plush': 15, 'rc_car': 40, 'rc_drone': 60,
    'board_game': 25, 'card_game': 15, 'puzzle': 15,
    'ride_on': 200, 'power_wheels': 300, 'playhouse': 200,
    'art_supplies': 20, 'science_kit': 25, 'slime': 10,
    'nerf': 25, 'water_gun': 15, 'toy_vehicle': 20,
    'train_set': 50, 'play_kitchen': 100, 'play_tool': 25,
    'trampoline_toy': 100, 'swing': 60, 'outdoor_toy': 30,
    'baby_toy': 15, 'educational_toy': 20, 'robot_toy': 40,
    'magnet_tiles': 40, 'play_tent': 30,
}
_TOY_BRAND_FACTORS = {
    'lego': 1.40, 'playmobil': 1.10, 'barbie': 1.00,
    'hot wheels': 0.80, 'nerf': 1.00, 'fisher-price': 0.90,
    'melissa & doug': 1.00, 'magna-tiles': 1.50,
    'power wheels': 1.20, 'step2': 1.10, 'little tikes': 0.90,
    'nintendo': 1.30, 'pokemon': 1.10, 'disney': 1.10,
    'hasbro': 0.90, 'mattel': 0.90, 'vtech': 0.80,
    'leapfrog': 0.80, 'crayola': 0.70, 'play-doh': 0.70,
}

# ──── PET SUPPLIES ────
_PET_TYPE_PRICES = {
    'dog_food': 40, 'cat_food': 30, 'dog_treat': 15, 'cat_treat': 10,
    'dog_bed': 40, 'cat_bed': 30, 'dog_crate': 60, 'cat_tree': 60,
    'leash': 15, 'collar': 15, 'harness': 25, 'dog_toy': 12,
    'cat_toy': 8, 'aquarium': 80, 'fish_tank': 80, 'filter': 20,
    'bird_cage': 60, 'litter_box': 25, 'cat_litter': 20,
    'pet_carrier': 40, 'grooming': 20, 'flea_treatment': 30,
    'pet_gate': 35, 'pet_camera': 50, 'pet_fountain': 30,
    'dog_door': 40, 'training_pad': 20, 'pet_ramp': 60,
    'pet_stroller': 80,
}

# ──── OFFICE PRODUCTS ────
_OFFICE_TYPE_PRICES = {
    'printer': 150, 'scanner': 100, 'shredder': 80, 'laminator': 40,
    'label_maker': 30, 'calculator': 15, 'desk_organizer': 20,
    'filing_cabinet': 100, 'whiteboard': 40, 'chair_office': 200,
    'desk_office': 200, 'monitor_stand': 25, 'keyboard_tray': 30,
    'paper': 25, 'ink_cartridge': 30, 'toner': 40,
    'pens': 8, 'pencils': 5, 'markers': 10, 'highlighters': 6,
    'notebooks': 8, 'planner': 15, 'sticky_notes': 5,
    'tape': 5, 'glue': 4, 'stapler': 10, 'scissors': 8,
    'envelope': 10, 'binder': 8, 'folder': 6,
    'desk_pad': 15, 'mouse_pad': 12, 'webcam': 60,
    'headset_office': 40, 'usb_hub': 25, 'docking_station': 100,
}

# ──── BOOKS & MEDIA ────
_BOOK_TYPE_PRICES = {
    'hardcover': 25, 'paperback': 15, 'textbook': 50,
    'art_book': 30, 'cookbook': 25, 'childrens_book': 10,
    'comic': 12, 'manga': 10, 'audiobook': 20,
    'dvd': 15, 'blu_ray': 20, 'vinyl_record': 25,
    'cd': 12, 'calendar': 12, 'journal': 12,
    'coloring_book': 8, 'sticker_book': 6,
    'art_supplies_media': 20, 'craft_supplies': 15,
}

# ──── HEALTH & HOUSEHOLD ────
_HEALTH_TYPE_PRICES = {
    'vitamin': 20, 'supplement': 25, 'protein_powder': 35,
    'first_aid': 15, 'thermometer': 20, 'blood_pressure': 40,
    'pulse_oximeter': 25, 'nebulizer': 50, 'hearing_aid': 200,
    'cpap': 300, 'wheelchair': 200, 'walker_medical': 60,
    'knee_brace': 25, 'back_brace': 30, 'compression': 15,
    'essential_oil': 15, 'diffuser': 25, 'humidifier_health': 50,
    'air_purifier_health': 150, 'water_filter': 30,
    'cleaning_supplies': 12, 'detergent': 15, 'trash_bags': 12,
    'paper_towels': 15, 'toilet_paper': 15, 'tissues': 8,
    'hand_soap': 8, 'hand_sanitizer': 8, 'mask': 10,
    'gloves_medical': 12, 'bandage': 8,
    'battery': 12, 'plastic_bags': 5, 'kitchen_wrap': 5,
    'sponges': 5, 'air_freshener': 6,
}


# ================================================================
# PRODUCT TYPE ESTIMATOR FUNCTIONS
# ================================================================

def _estimate_laptop_price(title, brand, specs, flags):
    """Estimate a realistic laptop price from specs."""
    t = title.lower()
    cpu_tier = specs.get('_cpu_tier')
    generation = specs.get('_generation')
    ram = specs.get('ram_gb', 0)
    storage = specs.get('storage_gb', 0)
    storage_tb = specs.get('storage_tb', 0)

    base = _CPU_BASE_PRICES.get(cpu_tier, 500) if cpu_tier else 500

    if generation and cpu_tier and cpu_tier.startswith('i'):
        current_gen = 14
        age = max(0, current_gen - generation)
        depreciation = max(0.35, 1.0 - age * 0.08)
        base *= depreciation

    if ram >= 64:   base *= 1.35
    elif ram >= 32: base *= 1.20
    elif ram >= 16: base *= 1.05
    elif 0 < ram <= 4: base *= 0.75

    if storage_tb >= 2:             base *= 1.20
    elif storage_tb >= 1 or storage >= 1024: base *= 1.12
    elif storage >= 512:            base *= 1.05
    elif 0 < storage <= 128:        base *= 0.90

    screen = specs.get('screen_inch', 0)
    if screen >= 17:   base *= 1.15
    elif screen >= 15: base *= 1.05

    base *= _LAPTOP_BRAND_FACTORS.get(brand, 0.95)

    if flags.get('is_premium', 0): base *= 1.10
    if flags.get('is_budget', 0):  base *= 0.80
    if re.search(r'\b(gaming|gamer|rtx|gtx|geforce|radeon)\b', t): base *= 1.20
    if re.search(r'\b(refurbished|renewed|refurb|used|pre-owned)\b', t): base *= 0.60

    return round(base, 2)


def _estimate_phone_price(title, brand, specs, flags):
    """Estimate a realistic phone price from specs."""
    t = title.lower()
    storage = specs.get('storage_gb', 0)

    base = _PHONE_BRAND_PRICES.get(brand, 350)

    if storage >= 1024:   base *= 1.40
    elif storage >= 512:  base *= 1.25
    elif storage >= 256:  base *= 1.10
    elif storage >= 128:  base *= 1.00
    elif 0 < storage <= 64: base *= 0.80

    if re.search(r'\b(ultra|pro max)\b', t):   base *= 1.50
    elif re.search(r'\b(pro|plus|\+)\b', t):   base *= 1.25
    if flags.get('is_budget', 0): base *= 0.60
    if re.search(r'\b(refurbished|renewed|used|pre-owned)\b', t): base *= 0.55

    return round(base, 2)


def _estimate_tv_price(title, brand, specs, flags):
    """Estimate a realistic TV price from specs."""
    t = title.lower()
    screen = specs.get('screen_inch', 0)

    # ── Streaming sticks / media players are NOT TVs ──
    if re.search(r'\b(fire tv stick|firestick|roku stick|roku express|chromecast|streaming stick|apple tv|nvidia shield|streaming device|streaming player|media player|mi box|mi stick)\b', t):
        base = 45
        if re.search(r'\b(4k|uhd)\b', t): base = 55
        if re.search(r'\b(max|cube|ultra|pro)\b', t): base *= 1.20
        return round(base, 2)

    # ── DVD/Blu-ray players ──
    if re.search(r'\b(dvd player|blu-?ray player|disc player)\b', t):
        return round(80, 2)

    base = 400
    for (lo, hi), price in _TV_SIZE_PRICES.items():
        if lo <= screen <= hi:
            base = price
            break

    if re.search(r'\b(oled)\b', t):                      base *= 1.60
    elif re.search(r'\b(qled|neo qled|mini.?led)\b', t):  base *= 1.30
    elif re.search(r'\b(4k|uhd)\b', t):                   base *= 1.10

    premium_tv = {'sony': 1.15, 'lg': 1.10, 'samsung': 1.10}
    base *= premium_tv.get(brand, 0.90)
    if re.search(r'\b(smart|roku|fire|android)\b', t): base *= 1.05

    return round(base, 2)


def _estimate_audio_price(title, brand, specs, flags):
    """Estimate audio product price."""
    t = title.lower()

    # Detect audio sub-type
    if re.search(r'\b(earbuds?|ear buds?|in-?ear|airpods|galaxy buds)\b', t):
        atype = 'earbuds'
    elif re.search(r'\b(headphones?|headset|over-?ear|on-?ear)\b', t):
        atype = 'headphones_over'
    elif re.search(r'\b(soundbar|sound bar)\b', t):
        atype = 'soundbar'
    elif re.search(r'\b(speaker)\b', t):
        if re.search(r'\b(portable|bluetooth|mini|small)\b', t):
            atype = 'speaker_portable'
        else:
            atype = 'speaker_home'
    elif re.search(r'\b(microphone|mic)\b', t):
        atype = 'microphone'
    elif re.search(r'\b(turntable|record player)\b', t):
        atype = 'turntable'
    elif re.search(r'\b(receiver|amplifier|amp)\b', t):
        atype = 'receiver'
    elif re.search(r'\b(karaoke)\b', t):
        atype = 'karaoke'
    else:
        atype = 'headphones_over'

    base = _AUDIO_TYPE_PRICES.get(atype, 80)
    base *= _AUDIO_BRAND_FACTORS.get(brand, 0.90)

    if re.search(r'\b(noise cancel|anc|active noise)\b', t): base *= 1.40
    if re.search(r'\b(dolby atmos|surround|5\.1|7\.1)\b', t): base *= 1.50
    if flags.get('is_wireless', 0): base *= 1.10
    if flags.get('is_premium', 0):  base *= 1.25
    if flags.get('is_budget', 0):   base *= 0.60
    if re.search(r'\b(gaming)\b', t): base *= 1.15
    if re.search(r'\b(studio|professional|pro)\b', t): base *= 1.30
    if re.search(r'\b(refurbished|renewed)\b', t): base *= 0.60

    return round(base, 2)


def _estimate_camera_price(title, brand, specs, flags):
    """Estimate camera/photo product price."""
    t = title.lower()

    # ── IMPORTANT: detect accessories BEFORE camera types ──
    # Otherwise "GoPro Hero 13 Protective Housing" matches 'gopro' → action_cam ($300)
    if re.search(r'\b(battery|np-f|np-w|lp-e|en-el|bp-)\b', t):
        return round(25 * (1.3 if brand in ('sony','canon','nikon','fujifilm') else 0.8), 2)
    elif re.search(r'\b(housing|underwater|waterproof case|protective case|protective housing)\b', t):
        return round(35, 2)
    elif re.search(r'\b(propeller|prop guard|propeller guard|landing gear)\b', t):
        return round(20, 2)
    elif re.search(r'\b(gimbal|stabilizer)\b', t) and not re.search(r'\b(drone|quadcopter|mavic|mini)\b.*\b(pro|se|air)\b', t):
        return round(120, 2)  # standalone gimbal, not a drone WITH gimbal
    elif re.search(r'\b(nd filter|uv filter|cpl filter|filter kit)\b', t):
        return round(30, 2)
    elif re.search(r'\b(camera bag|camera backpack|camera strap|camera sling)\b', t):
        return round(40, 2)
    elif re.search(r'\b(lens)\b', t) and not re.search(r'\b(camera|body|kit)\b', t):
        # Standalone lens (not a camera kit with lens)
        if re.search(r'\b(telephoto|70.?200|100.?400|200.?600)\b', t): return round(800, 2)
        elif re.search(r'\b(wide.?angle|16.?35|14.?24)\b', t): return round(600, 2)
        elif re.search(r'\b(macro|90mm|100mm)\b', t): return round(400, 2)
        elif re.search(r'\b(50mm|35mm|85mm|prime)\b', t): return round(200, 2)
        elif re.search(r'\b(zoom|24.?70|24.?105)\b', t): return round(500, 2)
        else: return round(250, 2)
    elif re.search(r'\b(tripod|softbox|light|backdrop|reflector|lens cloth|lens cap)\b', t):
        return round(40 * (0.80 if flags.get('is_budget', 0) else 1.0), 2)

    # ── Camera body types ──
    if re.search(r'\b(mirrorless)\b', t):       ctype = 'mirrorless'
    elif re.search(r'\b(dslr)\b', t):            ctype = 'dslr'
    elif re.search(r'\b(drone)\b', t):           ctype = 'drone'
    elif re.search(r'\b(gopro|action cam)\b', t): ctype = 'action_cam'
    elif re.search(r'\b(webcam)\b', t):          ctype = 'webcam'
    elif re.search(r'\b(dash cam)\b', t):        ctype = 'dash_cam'
    elif re.search(r'\b(security|surveillance)\b', t): ctype = 'security_cam'
    elif re.search(r'\b(instant|instax|polaroid)\b', t): ctype = 'instant'
    elif re.search(r'\b(trail cam|game cam)\b', t): ctype = 'trail_cam'
    else:
        ctype = 'point_shoot'

    base = _CAMERA_TYPE_PRICES.get(ctype, 200)
    base *= _CAMERA_BRAND_FACTORS.get(brand, 0.90)

    if re.search(r'\b(4k)\b', t): base *= 1.15
    if re.search(r'\b(8k)\b', t): base *= 1.40
    if flags.get('is_premium', 0): base *= 1.25
    if re.search(r'\b(refurbished|renewed)\b', t): base *= 0.55

    return round(base, 2)


def _estimate_gaming_price(title, brand, specs, flags):
    """Estimate gaming product price."""
    t = title.lower()

    # ── IMPORTANT: detect accessories/games BEFORE console keywords ──
    # Otherwise "PS5 Pulse 3D Headset" matches 'ps5' → console ($450)
    _game_kw = ['game', 'edition', 'deluxe edition', 'standard edition',
                'digital code', 'season pass', 'dlc', 'expansion',
                'legacy', 'hogwarts', 'starfield', 'mario kart',
                'god of war', 'spider-man', 'zelda', 'final fantasy']
    _gaming_acc_kw = ['controller', 'gamepad', 'joystick', 'manette', 'mando',
                      'dualsense', 'dualshock', 'joy-con', 'joycon',
                      'headset', 'earbuds', 'headphones',
                      'charging station', 'charging dock', 'dock set',
                      'stand', 'vertical stand',
                      'camera', 'hd camera',
                      'remote', 'media remote', 'remote control',
                      'cover plate', 'skin', 'faceplate', 'decal',
                      'case', 'carrying case', 'travel case',
                      'screen protector', 'tempered glass',
                      'cable', 'hdmi', 'usb', 'charger', 'adapter',
                      'thumb grip', 'thumbstick', 'trigger',
                      'steering wheel', 'racing wheel', 'flight stick',
                      'game pass', 'subscription', 'membership',
                      'gift card', 'prepaid']

    is_game = any(kw in t for kw in _game_kw)
    is_gaming_acc = any(kw in t for kw in _gaming_acc_kw)

    if is_gaming_acc or is_game:
        # It's NOT a console — detect the specific accessory type
        if re.search(r'\b(controller|gamepad|joystick|dualsense|dualshock|joy.?con|manette|mando)\b', t):
            gtype = 'controller'
        elif re.search(r'\b(headset|headphones?|earbuds?)\b', t):
            gtype = 'headset'
        elif re.search(r'\b(steering wheel|racing wheel|flight stick)\b', t):
            gtype = 'steering_wheel'
        elif re.search(r'\b(game pass|subscription|membership|gift card|prepaid)\b', t):
            gtype = 'video_game'  # digital subscription ~$30-60
        elif re.search(r'\b(dock set|dock|docking)\b', t):
            gtype = 'dock'
        elif re.search(r'\b(stand|vertical stand|charging dock|charging station)\b', t):
            gtype = 'stand'
        elif is_game:
            gtype = 'video_game'
        else:
            gtype = 'controller'  # generic gaming accessory
    elif re.search(r'\b(vr|virtual reality|quest|psvr)\b', t):
        gtype = 'vr_headset'
    elif re.search(r'\b(board game|tabletop)\b', t):
        gtype = 'board_game'
    elif re.search(r'\b(card game|trading card|pokemon card)\b', t):
        gtype = 'card_game'
    elif re.search(r'\b(sd card|micro sd|memory card)\b', t):
        gtype = 'memory_card'
    elif re.search(r'\b(gaming keyboard|mechanical keyboard)\b', t):
        gtype = 'keyboard'
    elif re.search(r'\b(gaming mouse)\b', t):
        gtype = 'mouse'
    elif re.search(r'\b(gaming chair)\b', t):
        gtype = 'chair'
    elif re.search(r'\b(gaming monitor|monitor)\b', t):
        gtype = 'monitor'
    elif re.search(r'\b(capture card)\b', t):
        gtype = 'capture_card'
    elif re.search(r'\b(playstation|ps5|ps4|xbox|nintendo switch|console)\b', t):
        gtype = 'console'
    else:
        gtype = 'video_game'

    base = _GAMING_TYPE_PRICES.get(gtype, 40)

    # Console brand adjustment
    if gtype == 'console':
        if re.search(r'\b(ps5|playstation 5)\b', t):   base = 450
        elif re.search(r'\b(xbox series x)\b', t):      base = 450
        elif re.search(r'\b(xbox series s)\b', t):       base = 280
        elif re.search(r'\b(nintendo switch oled)\b', t): base = 350
        elif re.search(r'\b(nintendo switch)\b', t):     base = 300
        elif re.search(r'\b(steam deck)\b', t):          base = 400

    # Controller brand adjustment
    if gtype == 'controller':
        if 'elite' in t: base = 160
        elif 'dualsense edge' in t: base = 180
        elif 'pro controller' in t: base = 65

    if flags.get('is_premium', 0): base *= 1.15
    if re.search(r'\b(refurbished|renewed)\b', t): base *= 0.60
    storage = specs.get('storage_gb', 0)
    if gtype == 'memory_card' and storage > 0:
        base = 10 + storage * 0.08  # ~$10 base + per-GB

    return round(base, 2)


def _estimate_wearable_price(title, brand, specs, flags):
    """Estimate wearable/smartwatch price."""
    t = title.lower()
    base = _WEARABLE_BRAND_PRICES.get(brand, 100)

    if re.search(r'\b(ultra|pro|premium|titanium|sapphire)\b', t): base *= 1.50
    if re.search(r'\b(se|lite|basic)\b', t): base *= 0.60
    if re.search(r'\b(fitness|band|tracker)\b', t) and not re.search(r'\b(watch)\b', t):
        base *= 0.50
    if re.search(r'\b(refurbished|renewed)\b', t): base *= 0.55

    return round(base, 2)


def _estimate_smarthome_price(title, brand, specs, flags):
    """Estimate smart home product price."""
    t = title.lower()

    if re.search(r'\b(thermostat)\b', t):        stype = 'thermostat'
    elif re.search(r'\b(doorbell)\b', t):         stype = 'doorbell'
    elif re.search(r'\b(security cam|indoor cam|outdoor cam)\b', t): stype = 'security_cam'
    elif re.search(r'\b(smart display|echo show|nest hub)\b', t): stype = 'display'
    elif re.search(r'\b(smart speaker|echo|homepod|google home)\b', t): stype = 'speaker'
    elif re.search(r'\b(smart plug)\b', t):       stype = 'plug'
    elif re.search(r'\b(smart switch)\b', t):     stype = 'switch'
    elif re.search(r'\b(smart bulb|smart light|philips hue)\b', t): stype = 'bulb'
    elif re.search(r'\b(smart lock|deadbolt)\b', t): stype = 'lock'
    elif re.search(r'\b(robot vacuum|roomba)\b', t): stype = 'vacuum_robot'
    elif re.search(r'\b(air purifier)\b', t):     stype = 'air_purifier'
    elif re.search(r'\b(humidifier)\b', t):       stype = 'humidifier'
    elif re.search(r'\b(heater|space heater)\b', t): stype = 'heater'
    else:
        stype = 'plug'

    base = _SMARTHOME_TYPE_PRICES.get(stype, 50)

    premium_sh = {'nest': 1.20, 'ring': 1.10, 'ecobee': 1.15, 'philips': 1.10,
                   'irobot': 1.30, 'dyson': 1.80, 'amazon': 0.90, 'google': 1.00}
    base *= premium_sh.get(brand, 0.90)

    if flags.get('is_smart', 0): base *= 1.05
    if flags.get('is_premium', 0): base *= 1.20

    return round(base, 2)


def _estimate_clothing_price(title, brand, specs, flags):
    """Estimate clothing & accessories price."""
    t = title.lower()

    # Detect clothing sub-type
    if re.search(r'\b(running shoe|basketball shoe|training shoe|athletic shoe|sneaker|air jordan|air max|air force|ultraboost|ultra boost)\b', t):
        ctype = 'shoes_athletic'
    elif re.search(r'\b(boot|timberland|dr marten|hiking boot|work boot|rain boot|snow boot|chelsea boot|combat boot)\b', t):
        ctype = 'shoes_boots'
    elif re.search(r'\b(sandal|slide|flip flop|birkenstock|crocs)\b', t):
        ctype = 'shoes_sandals'
    elif re.search(r'\b(shoe|loafer|oxford|derby|heel|pump|flat)\b', t):
        ctype = 'shoes_casual'
    elif re.search(r'\b(jacket|parka|windbreaker|raincoat|puffer|down jacket|fleece|vest)\b', t):
        ctype = 'jacket'
    elif re.search(r'\b(coat|trench|overcoat|peacoat|winter coat)\b', t):
        ctype = 'coat'
    elif re.search(r'\b(jeans|denim)\b', t):
        ctype = 'jeans'
    elif re.search(r'\b(dress pants|chinos|khakis|trousers|pants|legging|yoga pants)\b', t):
        ctype = 'pants'
    elif re.search(r'\b(dress)\b', t) and not re.search(r'\b(dress shirt|dress shoe)\b', t):
        ctype = 'dress'
    elif re.search(r'\b(suit|blazer|tuxedo)\b', t):
        ctype = 'suit'
    elif re.search(r'\b(hoodie|sweatshirt|pullover)\b', t):
        ctype = 'hoodie'
    elif re.search(r'\b(sweater|cardigan|turtleneck)\b', t):
        ctype = 'sweater'
    elif re.search(r'\b(t-?shirt|tee)\b', t):
        ctype = 'tshirt'
    elif re.search(r'\b(shirt|polo|button.?down|blouse)\b', t):
        ctype = 'shirt'
    elif re.search(r'\b(shorts)\b', t):
        ctype = 'shorts'
    elif re.search(r'\b(skirt)\b', t):
        ctype = 'skirt'
    elif re.search(r'\b(watch)\b', t):
        ctype = 'watch'
    elif re.search(r'\b(sunglasses|eyeglasses)\b', t):
        ctype = 'sunglasses'
    elif re.search(r'\b(handbag|purse|tote|clutch|crossbody)\b', t):
        ctype = 'handbag'
    elif re.search(r'\b(backpack|rucksack)\b', t):
        ctype = 'backpack'
    elif re.search(r'\b(belt)\b', t):
        ctype = 'belt'
    elif re.search(r'\b(hat|cap|beanie)\b', t):
        ctype = 'hat'
    elif re.search(r'\b(scarf|shawl)\b', t):
        ctype = 'scarf'
    elif re.search(r'\b(glove)\b', t):
        ctype = 'gloves'
    elif re.search(r'\b(swimsuit|bikini|swim trunk)\b', t):
        ctype = 'swimwear'
    elif re.search(r'\b(pajama|sleepwear|nightgown|robe)\b', t):
        ctype = 'pajamas'
    elif re.search(r'\b(underwear|boxer|brief|bra|panty|lingerie)\b', t):
        ctype = 'underwear'
    elif re.search(r'\b(sock)\b', t):
        ctype = 'socks'
    elif re.search(r'\b(wallet)\b', t):
        ctype = 'belt'  # ~$30 similar price tier
    else:
        ctype = 'shirt'

    base = _CLOTHING_TYPE_PRICES.get(ctype, 35)
    base *= _CLOTHING_BRAND_FACTORS.get(brand, 0.90)

    # Ultra-luxury brand detection (not in _CLOTHING_BRAND_FACTORS)
    t = title.lower()
    if re.search(r'\b(rolex|omega|cartier|patek philippe|audemars piguet|tag heuer|breitling|iwc|jaeger)\b', t):
        if ctype == 'watch': base = max(base, 5000)
        else: base *= 5.0
    elif re.search(r'\b(louis vuitton|chanel|hermes|hermès|dior|versace|fendi|balenciaga|valentino|givenchy|saint laurent|ysl|bottega veneta)\b', t):
        base *= 5.0
    elif re.search(r'\b(canada goose)\b', t):
        # Already has a brand factor of 3.0, but for specific items boost more
        if ctype in ('jacket', 'coat'): base = max(base, 800)

    if flags.get('is_premium', 0): base *= 1.20
    if flags.get('is_budget', 0):  base *= 0.60

    # Pack multiplier (socks, underwear, t-shirts)
    pack = specs.get('pack_qty', 1)
    if pack > 1 and ctype in ('socks', 'underwear', 'tshirt'):
        # Multi-pack of cheap items: $3-5 per unit × pack
        per_unit = 3.50 if brand in ('hanes', 'fruit of the loom', 'gildan') else 5.0
        base = per_unit * pack

    return round(base, 2)


def _estimate_appliance_price(title, brand, specs, flags):
    """Estimate home appliance price."""
    t = title.lower()

    # Detect appliance sub-type
    if re.search(r'\b(blender|nutribullet)\b', t):         atype = 'blender'
    elif re.search(r'\b(stand mixer|mixer)\b', t):          atype = 'mixer'
    elif re.search(r'\b(food processor)\b', t):             atype = 'food_processor'
    elif re.search(r'\b(toaster)\b', t):                    atype = 'toaster'
    elif re.search(r'\b(espresso|cappuccino)\b', t):        atype = 'espresso'
    elif re.search(r'\b(coffee maker|coffee machine|keurig|nespresso)\b', t): atype = 'coffee_maker'
    elif re.search(r'\b(microwave)\b', t):                  atype = 'microwave'
    elif re.search(r'\b(air fryer)\b', t):                  atype = 'air_fryer'
    elif re.search(r'\b(slow cooker|crock.?pot)\b', t):     atype = 'slow_cooker'
    elif re.search(r'\b(pressure cooker|instant pot)\b', t): atype = 'pressure_cooker'
    elif re.search(r'\b(rice cooker)\b', t):                atype = 'rice_cooker'
    elif re.search(r'\b(juicer)\b', t):                     atype = 'juicer'
    elif re.search(r'\b(ice maker)\b', t):                  atype = 'ice_maker'
    elif re.search(r'\b(dishwasher)\b', t):                 atype = 'dishwasher'
    elif re.search(r'\b(refrigerator|fridge)\b', t):        atype = 'refrigerator'
    elif re.search(r'\b(washer|washing machine)\b', t):     atype = 'washer'
    elif re.search(r'\b(dryer)\b', t):                      atype = 'dryer'
    elif re.search(r'\b(oven|range|stove)\b', t):           atype = 'oven'
    elif re.search(r'\b(vacuum|dyson v\d|shark|roomba)\b', t): atype = 'vacuum'
    elif re.search(r'\b(iron|steamer)\b', t):               atype = 'iron'
    elif re.search(r'\b(sewing machine)\b', t):             atype = 'sewing_machine'
    elif re.search(r'\b(waffle maker|waffle iron)\b', t):   atype = 'waffle_maker'
    elif re.search(r'\b(griddle|grill pan)\b', t):          atype = 'griddle'
    elif re.search(r'\b(kettle)\b', t):                     atype = 'kettle'
    elif re.search(r'\b(dehydrator)\b', t):                 atype = 'dehydrator'
    elif re.search(r'\b(pots? and pans?|cookware set|pan set|pot set)\b', t): atype = 'cookware_set'
    elif re.search(r'\b(knife set|knives)\b', t):           atype = 'knife_set'
    elif re.search(r'\b(cutting board)\b', t):              atype = 'cutting_board'
    elif re.search(r'\b(bakeware|baking|pie pan|cake pan|cookie sheet|muffin)\b', t): atype = 'bakeware'
    elif re.search(r'\b(pan|skillet|wok|dutch oven|saucepan|stockpot)\b', t): atype = 'pan_set'
    else:
        atype = 'blender'

    base = _APPLIANCE_TYPE_PRICES.get(atype, 60)
    base *= _APPLIANCE_BRAND_FACTORS.get(brand, 0.90)

    if flags.get('is_premium', 0): base *= 1.20
    if flags.get('is_budget', 0):  base *= 0.60

    # Set/count multiplier
    pack = specs.get('pack_qty', 1)
    if pack > 1 and atype in ('bakeware', 'pan_set', 'pot_set', 'knife_set'):
        base *= min(pack * 0.50, 3.0)

    return round(base, 2)


def _estimate_furniture_price(title, brand, specs, flags):
    """Estimate furniture price."""
    t = title.lower()

    if re.search(r'\b(mattress topper|mattress pad)\b', t):   ftype = 'mattress_topper'
    elif re.search(r'\b(mattress)\b', t):                      ftype = 'mattress'
    elif re.search(r'\b(bed frame|platform bed|headboard)\b', t): ftype = 'bed_frame'
    elif re.search(r'\b(bunk bed)\b', t):                      ftype = 'bunk_bed'
    elif re.search(r'\b(crib)\b', t):                          ftype = 'crib'
    elif re.search(r'\b(sectional)\b', t):                     ftype = 'sectional'
    elif re.search(r'\b(loveseat)\b', t):                      ftype = 'loveseat'
    elif re.search(r'\b(sofa|couch|futon)\b', t):              ftype = 'sofa'
    elif re.search(r'\b(recliner)\b', t):                      ftype = 'recliner'
    elif re.search(r'\b(office chair|ergonomic chair|gaming chair)\b', t): ftype = 'office_chair'
    elif re.search(r'\b(desk)\b', t):                          ftype = 'desk'
    elif re.search(r'\b(dining table)\b', t):                  ftype = 'dining_table'
    elif re.search(r'\b(dining chair|kitchen chair)\b', t):    ftype = 'dining_chair'
    elif re.search(r'\b(bookshelf|bookcase|shelf)\b', t):      ftype = 'bookshelf'
    elif re.search(r'\b(dresser|chest of drawers)\b', t):      ftype = 'dresser'
    elif re.search(r'\b(nightstand|bedside table)\b', t):      ftype = 'nightstand'
    elif re.search(r'\b(tv stand|media console|entertainment center)\b', t): ftype = 'tv_stand'
    elif re.search(r'\b(coffee table)\b', t):                  ftype = 'coffee_table'
    elif re.search(r'\b(end table|side table)\b', t):          ftype = 'end_table'
    elif re.search(r'\b(wardrobe|armoire|closet)\b', t):       ftype = 'wardrobe'
    elif re.search(r'\b(cabinet|pantry|cupboard)\b', t):       ftype = 'cabinet'
    elif re.search(r'\b(shoe rack|shoe organizer)\b', t):      ftype = 'shoe_rack'
    elif re.search(r'\b(bar stool|counter stool)\b', t):       ftype = 'bar_stool'
    elif re.search(r'\b(ottoman|storage bench)\b', t):         ftype = 'storage_ottoman'
    elif re.search(r'\b(bean bag)\b', t):                      ftype = 'bean_bag'
    elif re.search(r'\b(vanity)\b', t):                        ftype = 'vanity'
    elif re.search(r'\b(mirror)\b', t):                        ftype = 'mirror'
    elif re.search(r'\b(rug|carpet|area rug)\b', t):           ftype = 'rug'
    elif re.search(r'\b(curtain|drape|blind|shade)\b', t):     ftype = 'curtain'
    elif re.search(r'\b(pillow)\b', t):                        ftype = 'pillow'
    elif re.search(r'\b(comforter|duvet|quilt|bedding)\b', t): ftype = 'comforter'
    elif re.search(r'\b(changing table)\b', t):                ftype = 'changing_table'
    else:
        ftype = 'bookshelf'

    base = _FURNITURE_TYPE_PRICES.get(ftype, 100)
    base *= _FURNITURE_BRAND_FACTORS.get(brand, 0.90)

    # Size adjustments
    screen = specs.get('screen_inch', 0)  # reuse for height/size
    if ftype == 'rug':
        # Rug size from title e.g. "6x9", "8x10"
        rug_m = re.search(r"(\d+)'?\s*x\s*(\d+)'?", t)
        if rug_m:
            area = int(rug_m.group(1)) * int(rug_m.group(2))
            if area >= 80: base *= 2.0
            elif area >= 48: base *= 1.50
            elif area >= 20: base *= 1.10

    if re.search(r'\b(king)\b', t):       base *= 1.25
    elif re.search(r'\b(queen)\b', t):    base *= 1.10
    elif re.search(r'\b(full|double)\b', t): base *= 1.00
    elif re.search(r'\b(twin)\b', t):     base *= 0.80

    if flags.get('is_premium', 0): base *= 1.20
    if flags.get('is_budget', 0):  base *= 0.65

    if re.search(r'\b(led|rgb)\b', t): base *= 1.10
    if re.search(r'\b(with drawers|with storage)\b', t): base *= 1.15

    return round(base, 2)


def _estimate_decor_price(title, brand, specs, flags):
    """Estimate home decor price."""
    t = title.lower()

    if re.search(r'\b(rug|carpet|area rug)\b', t):      dtype = 'rug'
    elif re.search(r'\b(curtain|drape|blind|shade)\b', t): dtype = 'curtain'
    elif re.search(r'\b(pillow|cushion)\b', t):          dtype = 'pillow'
    elif re.search(r'\b(throw|blanket)\b', t):           dtype = 'throw_blanket'
    elif re.search(r'\b(wall art|canvas|painting|poster|print)\b', t): dtype = 'wall_art'
    elif re.search(r'\b(picture frame|photo frame)\b', t): dtype = 'picture_frame'
    elif re.search(r'\b(candle|diffuser|wax melt)\b', t): dtype = 'candle'
    elif re.search(r'\b(lamp|light|chandelier|pendant|sconce)\b', t):
        if re.search(r'\b(chandelier|pendant)\b', t):
            dtype = 'chandelier'
        else:
            dtype = 'lamp'
    elif re.search(r'\b(vase)\b', t):                    dtype = 'vase'
    elif re.search(r'\b(clock)\b', t):                   dtype = 'clock'
    elif re.search(r'\b(mirror)\b', t):                  dtype = 'mirror'
    elif re.search(r'\b(towel)\b', t):                   dtype = 'towel_set'
    elif re.search(r'\b(shower curtain)\b', t):          dtype = 'shower_curtain'
    elif re.search(r'\b(bath mat)\b', t):                dtype = 'bath_mat'
    elif re.search(r'\b(organizer|storage|basket|bin)\b', t): dtype = 'organizer'
    else:
        dtype = 'shelf_decor'

    base = _DECOR_TYPE_PRICES.get(dtype, 25)

    # Rug size adjustment
    if dtype == 'rug':
        rug_m = re.search(r"(\d+)'?\s*x\s*(\d+)'?", t)
        if rug_m:
            area = int(rug_m.group(1)) * int(rug_m.group(2))
            if area >= 80: base *= 2.50
            elif area >= 48: base *= 1.80
            elif area >= 20: base *= 1.20

    pack = specs.get('pack_qty', 1)
    if pack > 1: base *= min(pack * 0.65, 4.0)

    if flags.get('is_premium', 0): base *= 1.30

    return round(base, 2)


def _estimate_tools_price(title, brand, specs, flags):
    """Estimate tools & home improvement price."""
    t = title.lower()

    if re.search(r'\b(drill|driver)\b', t):            ttype = 'drill'
    elif re.search(r'\b(circular saw|miter saw|table saw|jigsaw|reciprocating saw|band saw|chainsaw)\b', t): ttype = 'saw'
    elif re.search(r'\b(sander)\b', t):                 ttype = 'sander'
    elif re.search(r'\b(grinder|angle grinder)\b', t):  ttype = 'grinder'
    elif re.search(r'\b(impact driver|impact wrench)\b', t): ttype = 'impact_driver'
    elif re.search(r'\b(wrench|socket set|ratchet)\b', t): ttype = 'wrench_set'
    elif re.search(r'\b(screwdriver)\b', t):            ttype = 'screwdriver_set'
    elif re.search(r'\b(pliers)\b', t):                 ttype = 'pliers_set'
    elif re.search(r'\b(hammer)\b', t):                 ttype = 'hammer'
    elif re.search(r'\b(pressure washer|power washer)\b', t): ttype = 'power_washer'
    elif re.search(r'\b(generator)\b', t):              ttype = 'generator'
    elif re.search(r'\b(air compressor|compressor)\b', t): ttype = 'air_compressor'
    elif re.search(r'\b(weld)\b', t):                   ttype = 'welding'
    elif re.search(r'\b(nail gun|nailer|brad nailer|finish nailer)\b', t): ttype = 'nail_gun'
    elif re.search(r'\b(router)\b', t):                 ttype = 'router'
    elif re.search(r'\b(workbench)\b', t):              ttype = 'workbench'
    elif re.search(r'\b(toolbox|tool box|tool bag|tool chest)\b', t): ttype = 'toolbox'
    elif re.search(r'\b(ladder|step stool)\b', t):      ttype = 'ladder'
    elif re.search(r'\b(paint sprayer|spray gun)\b', t): ttype = 'paint_sprayer'
    elif re.search(r'\b(heat gun)\b', t):               ttype = 'heat_gun'
    elif re.search(r'\b(oscillating tool|multi-?tool)\b', t): ttype = 'oscillating_tool'
    elif re.search(r'\b(tape measure|measuring)\b', t): ttype = 'tape_measure'
    elif re.search(r'\b(level)\b', t):                  ttype = 'level'
    elif re.search(r'\b(stud finder)\b', t):            ttype = 'stud_finder'
    elif re.search(r'\b(multimeter)\b', t):             ttype = 'multimeter'
    else:
        ttype = 'screwdriver_set'

    base = _TOOL_TYPE_PRICES.get(ttype, 30)
    base *= _TOOL_BRAND_FACTORS.get(brand, 0.85)

    if flags.get('is_premium', 0): base *= 1.20
    if flags.get('is_budget', 0):  base *= 0.65

    # Voltage hints for power tools
    volt_m = re.search(r'(\d+)\s*v\b', t)
    if volt_m:
        volts = int(volt_m.group(1))
        if volts >= 60:  base *= 1.30
        elif volts >= 40: base *= 1.15
        elif volts >= 20: base *= 1.00
        elif volts >= 12: base *= 0.80

    # Combo kits
    if re.search(r'\b(combo|kit|set)\b', t):
        pack = specs.get('pack_qty', 1)
        if pack >= 5: base *= 2.00
        elif pack >= 3: base *= 1.50
        elif pack >= 2: base *= 1.25

    return round(base, 2)


def _estimate_garden_price(title, brand, specs, flags):
    """Estimate garden & outdoor price."""
    t = title.lower()

    if re.search(r'\b(riding mower|lawn tractor|zero turn)\b', t): gtype = 'lawn_mower_riding'
    elif re.search(r'\b(lawn mower|mower|push mower)\b', t): gtype = 'lawn_mower'
    elif re.search(r'\b(trimmer|weed eater|edger)\b', t):    gtype = 'trimmer'
    elif re.search(r'\b(leaf blower|blower)\b', t):           gtype = 'leaf_blower'
    elif re.search(r'\b(chainsaw)\b', t):                     gtype = 'chainsaw'
    elif re.search(r'\b(hedge trimmer)\b', t):                gtype = 'hedge_trimmer'
    elif re.search(r'\b(pressure washer|power washer)\b', t): gtype = 'pressure_washer'
    elif re.search(r'\b(smoker)\b', t):                       gtype = 'smoker'
    elif re.search(r'\b(grill|bbq|barbecue)\b', t):           gtype = 'grill'
    elif re.search(r'\b(patio furniture|outdoor furniture|patio set|garden furniture)\b', t): gtype = 'patio_furniture'
    elif re.search(r'\b(patio umbrella|outdoor umbrella)\b', t): gtype = 'patio_umbrella'
    elif re.search(r'\b(fire pit|firepit)\b', t):             gtype = 'fire_pit'
    elif re.search(r'\b(outdoor light|solar light|string light|landscape light)\b', t): gtype = 'outdoor_light'
    elif re.search(r'\b(garden hose|hose)\b', t):             gtype = 'garden_hose'
    elif re.search(r'\b(sprinkler)\b', t):                    gtype = 'sprinkler'
    elif re.search(r'\b(shed|storage shed)\b', t):            gtype = 'shed'
    elif re.search(r'\b(greenhouse)\b', t):                   gtype = 'greenhouse'
    elif re.search(r'\b(hot tub|spa)\b', t):                  gtype = 'hot_tub'
    elif re.search(r'\b(pool|swimming pool)\b', t):           gtype = 'pool'
    elif re.search(r'\b(trampoline)\b', t):                   gtype = 'trampoline'
    elif re.search(r'\b(swing set|playset)\b', t):            gtype = 'swing_set'
    elif re.search(r'\b(snow blower|snowblower)\b', t):       gtype = 'snow_blower'
    elif re.search(r'\b(wheelbarrow)\b', t):                  gtype = 'wheelbarrow'
    elif re.search(r'\b(planter|flower pot|plant pot)\b', t): gtype = 'planter'
    elif re.search(r'\b(bird feeder)\b', t):                  gtype = 'bird_feeder'
    elif re.search(r'\b(seed|bulb)\b', t):                    gtype = 'seeds'
    elif re.search(r'\b(fertilizer|plant food)\b', t):        gtype = 'fertilizer'
    elif re.search(r'\b(soil|potting mix|compost)\b', t):     gtype = 'soil'
    elif re.search(r'\b(garden tool|pruner|shear|trowel|rake|shovel|hoe)\b', t): gtype = 'garden_tools'
    else:
        gtype = 'garden_tools'

    base = _GARDEN_TYPE_PRICES.get(gtype, 30)

    garden_brands = {'ego': 1.30, 'greenworks': 0.95, 'husqvarna': 1.30,
                     'stihl': 1.30, 'weber': 1.20, 'traeger': 1.30,
                     'big green egg': 2.50, 'kamado joe': 2.00,
                     'solo stove': 1.30, 'coleman': 0.80, 'ryobi': 0.80}
    base *= garden_brands.get(brand, 0.90)

    if re.search(r'\b(self.?propelled)\b', t): base *= 1.20
    if re.search(r'\b(electric|battery|cordless)\b', t): base *= 1.10
    if flags.get('is_premium', 0): base *= 1.20

    return round(base, 2)


def _estimate_sports_price(title, brand, specs, flags):
    """Estimate sports & outdoor equipment price."""
    t = title.lower()

    if re.search(r'\b(treadmill)\b', t):                     stype = 'treadmill'
    elif re.search(r'\b(exercise bike|spin bike|stationary bike|peloton)\b', t): stype = 'exercise_bike'
    elif re.search(r'\b(elliptical)\b', t):                   stype = 'elliptical'
    elif re.search(r'\b(rowing machine|rower)\b', t):         stype = 'rowing_machine'
    elif re.search(r'\b(weight bench|bench press)\b', t):     stype = 'weight_bench'
    elif re.search(r'\b(dumbbell set|dumbbell pair)\b', t):   stype = 'dumbbell_set'
    elif re.search(r'\b(dumbbell|weight)\b', t):              stype = 'dumbbell'
    elif re.search(r'\b(kettlebell)\b', t):                   stype = 'kettlebell'
    elif re.search(r'\b(barbell|weight plate)\b', t):         stype = 'barbell'
    elif re.search(r'\b(resistance band)\b', t):              stype = 'resistance_band'
    elif re.search(r'\b(yoga mat|exercise mat)\b', t):        stype = 'yoga_mat'
    elif re.search(r'\b(pull.?up bar|chin.?up bar)\b', t):    stype = 'pull_up_bar'
    elif re.search(r'\b(jump rope|skip rope)\b', t):          stype = 'jump_rope'
    elif re.search(r'\b(punching bag|heavy bag)\b', t):       stype = 'punching_bag'
    elif re.search(r'\b(boxing glove)\b', t):                 stype = 'boxing_gloves'
    elif re.search(r'\b(e-?bike|electric bike)\b', t):        stype = 'ebike'
    elif re.search(r'\b(bicycle|bike|cycling)\b', t) and not re.search(r'\b(exercise|spin|stationary)\b', t):
        stype = 'bicycle'
    elif re.search(r'\b(helmet)\b', t):                       stype = 'helmet'
    elif re.search(r'\b(ski|skiing)\b', t):                   stype = 'ski'
    elif re.search(r'\b(snowboard)\b', t):                    stype = 'snowboard'
    elif re.search(r'\b(surfboard)\b', t):                    stype = 'surfboard'
    elif re.search(r'\b(kayak|canoe)\b', t):                  stype = 'kayak'
    elif re.search(r'\b(tent)\b', t):                         stype = 'tent'
    elif re.search(r'\b(sleeping bag)\b', t):                 stype = 'sleeping_bag'
    elif re.search(r'\b(hiking boot)\b', t):                  stype = 'hiking_boots'
    elif re.search(r'\b(fishing rod|fishing pole)\b', t):     stype = 'fishing_rod'
    elif re.search(r'\b(fishing reel)\b', t):                 stype = 'fishing_reel'
    elif re.search(r'\b(golf club|golf iron|golf driver|putter)\b', t): stype = 'golf_club'
    elif re.search(r'\b(golf set|golf kit)\b', t):            stype = 'golf_set'
    elif re.search(r'\b(basketball)\b', t):                   stype = 'basketball'
    elif re.search(r'\b(football)\b', t):                     stype = 'football'
    elif re.search(r'\b(soccer ball)\b', t):                  stype = 'soccer_ball'
    elif re.search(r'\b(baseball bat)\b', t):                 stype = 'baseball_bat'
    elif re.search(r'\b(tennis racket)\b', t):                stype = 'tennis_racket'
    elif re.search(r'\b(scooter)\b', t):                      stype = 'scooter'
    elif re.search(r'\b(skateboard)\b', t):                   stype = 'skateboard'
    elif re.search(r'\b(massage gun|percussion)\b', t):       stype = 'massage_gun'
    elif re.search(r'\b(foam roller|roller)\b', t):           stype = 'foam_roller'
    elif re.search(r'\b(binocular)\b', t):                    stype = 'binoculars'
    elif re.search(r'\b(flashlight|lantern|headlamp)\b', t):  stype = 'flashlight'
    elif re.search(r'\b(cooler|ice chest)\b', t):             stype = 'cooler'
    elif re.search(r'\b(water bottle|hydro|tumbler)\b', t):   stype = 'water_bottle'
    elif re.search(r'\b(backpack)\b', t):                     stype = 'backpack_hiking'
    else:
        stype = 'yoga_mat'

    base = _SPORTS_TYPE_PRICES.get(stype, 40)
    base *= _SPORTS_BRAND_FACTORS.get(brand, 0.90)

    if flags.get('is_premium', 0): base *= 1.25
    if flags.get('is_budget', 0):  base *= 0.60

    return round(base, 2)


def _estimate_auto_price(title, brand, specs, flags):
    """Estimate automotive product price."""
    t = title.lower()

    if re.search(r'\b(dash cam|dashcam)\b', t):              atype = 'dash_cam'
    elif re.search(r'\b(jump starter|battery booster)\b', t): atype = 'jump_starter'
    elif re.search(r'\b(set of \d|4.?pack).*(tire)\b', t) or re.search(r'\b(tire).*(set of \d|4.?pack)\b', t):
        atype = 'tire_set'
    elif re.search(r'\b(tire|tyre)\b', t):                    atype = 'tire'
    elif re.search(r'\b(car cover|truck cover)\b', t):        atype = 'car_cover'
    elif re.search(r'\b(seat cover)\b', t):                   atype = 'seat_cover'
    elif re.search(r'\b(floor mat|cargo mat|trunk mat)\b', t): atype = 'floor_mat'
    elif re.search(r'\b(wiper|windshield wiper)\b', t):       atype = 'wiper'
    elif re.search(r'\b(headlight|tail light|fog light)\b', t): atype = 'headlight'
    elif re.search(r'\b(car battery|auto battery)\b', t):     atype = 'car_battery'
    elif re.search(r'\b(motor oil|engine oil|oil filter)\b', t): atype = 'oil'
    elif re.search(r'\b(air filter|cabin filter)\b', t):      atype = 'air_filter'
    elif re.search(r'\b(brake pad|brake rotor|brake kit)\b', t): atype = 'brake_pad'
    elif re.search(r'\b(car stereo|head unit|car radio)\b', t): atype = 'car_stereo'
    elif re.search(r'\b(subwoofer|sub)\b', t):                atype = 'subwoofer'
    elif re.search(r'\b(amplifier|amp)\b', t):                atype = 'amplifier'
    elif re.search(r'\b(roof rack|cargo rack)\b', t):         atype = 'roof_rack'
    elif re.search(r'\b(hitch|tow)\b', t):                    atype = 'hitch'
    elif re.search(r'\b(winch)\b', t):                        atype = 'winch'
    elif re.search(r'\b(jack|floor jack|jack stand)\b', t):   atype = 'jack'
    elif re.search(r'\b(air pump|tire inflator)\b', t):       atype = 'air_pump'
    elif re.search(r'\b(car wash|car soap|car shampoo)\b', t): atype = 'car_wash'
    elif re.search(r'\b(wax|polish|sealant)\b', t):           atype = 'wax'
    elif re.search(r'\b(obd|scanner|diagnostic)\b', t):       atype = 'obd_scanner'
    elif re.search(r'\b(steam cleaner)\b', t):                atype = 'steam_cleaner'
    elif re.search(r'\b(car vacuum|handheld vacuum)\b', t):   atype = 'car_vacuum'
    elif re.search(r'\b(phone mount|car mount)\b', t):        atype = 'phone_mount'
    elif re.search(r'\b(car charger)\b', t):                  atype = 'car_charger'
    else:
        atype = 'air_filter'

    base = _AUTO_TYPE_PRICES.get(atype, 30)

    auto_brands = {'vantrue': 1.20, 'noco': 1.15, 'thule': 1.40,
                   'yakima': 1.30, 'weathertech': 1.30, 'bosch': 1.10,
                   'pioneer': 1.10, 'kenwood': 1.10}
    base *= auto_brands.get(brand, 0.90)

    if flags.get('is_premium', 0): base *= 1.15

    return round(base, 2)


def _estimate_baby_price(title, brand, specs, flags):
    """Estimate baby product price."""
    t = title.lower()

    if re.search(r'\b(stroller|pushchair|jogger)\b', t):     btype = 'stroller'
    elif re.search(r'\b(car seat|booster seat)\b', t):        btype = 'car_seat'
    elif re.search(r'\b(high chair|highchair)\b', t):         btype = 'high_chair'
    elif re.search(r'\b(crib|cradle)\b', t):                  btype = 'crib'
    elif re.search(r'\b(bassinet)\b', t):                     btype = 'bassinet'
    elif re.search(r'\b(playpen|playard|play yard)\b', t):    btype = 'playpen'
    elif re.search(r'\b(baby monitor|video monitor)\b', t):   btype = 'baby_monitor'
    elif re.search(r'\b(baby swing|infant swing)\b', t):      btype = 'baby_swing'
    elif re.search(r'\b(bouncer|rocker)\b', t):               btype = 'bouncer'
    elif re.search(r'\b(walker)\b', t):                       btype = 'walker'
    elif re.search(r'\b(baby carrier|carrier)\b', t):         btype = 'carrier'
    elif re.search(r'\b(diaper bag)\b', t):                   btype = 'diaper_bag'
    elif re.search(r'\b(breast pump)\b', t):                  btype = 'breast_pump'
    elif re.search(r'\b(bottle|sippy cup)\b', t):             btype = 'bottle'
    elif re.search(r'\b(formula)\b', t):                      btype = 'formula'
    elif re.search(r'\b(diaper|nappy)\b', t):                 btype = 'diaper'
    elif re.search(r'\b(wipes)\b', t):                        btype = 'wipes'
    elif re.search(r'\b(pacifier|binky)\b', t):               btype = 'pacifier'
    elif re.search(r'\b(teether|teething)\b', t):             btype = 'teether'
    elif re.search(r'\b(bib)\b', t):                          btype = 'bib'
    elif re.search(r'\b(swaddle|sleep sack)\b', t):           btype = 'swaddle'
    elif re.search(r'\b(baby gate|safety gate)\b', t):        btype = 'baby_gate'
    elif re.search(r'\b(baby bath|infant tub)\b', t):         btype = 'baby_bath'
    elif re.search(r'\b(changing table|changing pad)\b', t):  btype = 'changing_table'
    else:
        btype = 'baby_toy'

    base = _BABY_TYPE_PRICES.get(btype, 25)
    base *= _BABY_BRAND_FACTORS.get(brand, 0.90)

    if flags.get('is_premium', 0): base *= 1.20

    pack = specs.get('pack_qty', 1)
    if pack > 1 and btype in ('bottle', 'diaper', 'wipes', 'bib', 'pacifier'):
        base *= min(pack * 0.55, 3.0)

    return round(base, 2)


def _estimate_beauty_price(title, brand, specs, flags):
    """Estimate beauty & personal care price."""
    t = title.lower()

    if re.search(r'\b(hair dryer|blow dryer)\b', t):         btype = 'hair_dryer'
    elif re.search(r'\b(hair styler|airwrap|curling wand|curling iron)\b', t): btype = 'hair_styler'
    elif re.search(r'\b(straightener|flat iron)\b', t):       btype = 'straightener'
    elif re.search(r'\b(hair clipper|hair trimmer)\b', t):    btype = 'hair_clipper'
    elif re.search(r'\b(beard trimmer)\b', t):                btype = 'beard_trimmer'
    elif re.search(r'\b(electric shaver|razor|shaver)\b', t): btype = 'electric_shaver'
    elif re.search(r'\b(electric toothbrush|oral-b|sonicare)\b', t): btype = 'electric_toothbrush'
    elif re.search(r'\b(water flosser|waterpik)\b', t):       btype = 'water_flosser'
    elif re.search(r'\b(perfume|eau de|fragrance|cologne)\b', t): btype = 'perfume'
    elif re.search(r'\b(serum|retinol|vitamin c serum|hyaluronic)\b', t): btype = 'serum'
    elif re.search(r'\b(moisturizer|cream|lotion|body butter)\b', t): btype = 'moisturizer'
    elif re.search(r'\b(sunscreen|spf)\b', t):                btype = 'sunscreen'
    elif re.search(r'\b(face wash|cleanser|face scrub)\b', t): btype = 'face_wash'
    elif re.search(r'\b(shampoo)\b', t):                      btype = 'shampoo'
    elif re.search(r'\b(conditioner)\b', t):                  btype = 'conditioner'
    elif re.search(r'\b(body wash|shower gel)\b', t):         btype = 'body_wash'
    elif re.search(r'\b(foundation|concealer|bb cream|cc cream)\b', t): btype = 'foundation'
    elif re.search(r'\b(mascara)\b', t):                      btype = 'mascara'
    elif re.search(r'\b(lipstick|lip gloss|lip balm)\b', t):  btype = 'lipstick'
    elif re.search(r'\b(eyeshadow|eye shadow|palette)\b', t): btype = 'eyeshadow'
    elif re.search(r'\b(makeup brush|brush set)\b', t):       btype = 'makeup_brush'
    elif re.search(r'\b(nail polish|nail art)\b', t):         btype = 'nail_polish'
    elif re.search(r'\b(nail drill)\b', t):                   btype = 'massage_device'
    elif re.search(r'\b(led mask|light therapy)\b', t):       btype = 'led_mask'
    elif re.search(r'\b(scale|bathroom scale)\b', t):         btype = 'scale'
    elif re.search(r'\b(deodorant|antiperspirant)\b', t):     btype = 'deodorant'
    elif re.search(r'\b(face mask|sheet mask|peel)\b', t):    btype = 'face_mask'
    elif re.search(r'\b(hair mask|hair treatment|hair oil)\b', t): btype = 'hair_mask'
    else:
        btype = 'face_wash'

    base = _BEAUTY_TYPE_PRICES.get(btype, 18)
    base *= _BEAUTY_BRAND_FACTORS.get(brand, 0.85)

    if flags.get('is_premium', 0): base *= 1.30

    pack = specs.get('pack_qty', 1)
    if pack > 1 and btype in ('face_mask', 'nail_polish', 'lipstick'):
        base *= min(pack * 0.60, 3.0)

    return round(base, 2)


def _estimate_music_price(title, brand, specs, flags):
    """Estimate musical instrument price."""
    t = title.lower()

    if re.search(r'\b(electric guitar)\b', t):           mtype = 'electric_guitar'
    elif re.search(r'\b(acoustic guitar|classical guitar)\b', t): mtype = 'acoustic_guitar'
    elif re.search(r'\b(bass guitar|electric bass)\b', t): mtype = 'bass_guitar'
    elif re.search(r'\b(ukulele|uke)\b', t):              mtype = 'ukulele'
    elif re.search(r'\b(violin|viola)\b', t):             mtype = 'violin'
    elif re.search(r'\b(cello)\b', t):                    mtype = 'cello'
    elif re.search(r'\b(flute)\b', t):                    mtype = 'flute'
    elif re.search(r'\b(trumpet)\b', t):                  mtype = 'trumpet'
    elif re.search(r'\b(saxophone|sax)\b', t):            mtype = 'saxophone'
    elif re.search(r'\b(clarinet)\b', t):                 mtype = 'clarinet'
    elif re.search(r'\b(drum kit|drum set|electronic drum)\b', t): mtype = 'drum_kit'
    elif re.search(r'\b(drum pad|practice pad)\b', t):    mtype = 'drum_pad'
    elif re.search(r'\b(cymbal)\b', t):                   mtype = 'cymbal'
    elif re.search(r'\b(digital piano|electric piano|stage piano)\b', t): mtype = 'digital_piano'
    elif re.search(r'\b(keyboard|synth)\b', t):           mtype = 'keyboard'
    elif re.search(r'\b(midi controller|midi)\b', t):     mtype = 'midi_controller'
    elif re.search(r'\b(audio interface)\b', t):          mtype = 'audio_interface'
    elif re.search(r'\b(guitar amp|bass amp|amplifier|combo amp)\b', t): mtype = 'guitar_amp'
    elif re.search(r'\b(pedal|effects? pedal|pedalboard)\b', t): mtype = 'pedal'
    elif re.search(r'\b(microphone|mic)\b', t):           mtype = 'microphone_music'
    elif re.search(r'\b(guitar string|bass string)\b', t): mtype = 'guitar_strings'
    elif re.search(r'\b(drum stick)\b', t):               mtype = 'drum_sticks'
    elif re.search(r'\b(tuner)\b', t):                    mtype = 'tuner'
    elif re.search(r'\b(capo)\b', t):                     mtype = 'capo'
    elif re.search(r'\b(music stand|sheet stand)\b', t):  mtype = 'music_stand'
    elif re.search(r'\b(strap)\b', t):                    mtype = 'strap'
    elif re.search(r'\b(bench|stool|throne)\b', t):       mtype = 'music_stand'  # ~$20
    elif re.search(r'\b(mute|mute pad|dampener)\b', t):   mtype = 'drum_sticks'
    else:
        mtype = 'acoustic_guitar'

    base = _MUSIC_TYPE_PRICES.get(mtype, 100)
    base *= _MUSIC_BRAND_FACTORS.get(brand, 0.85)

    if flags.get('is_premium', 0): base *= 1.25
    if flags.get('is_budget', 0):  base *= 0.55

    pack = specs.get('pack_qty', 1)
    if pack > 1 and mtype in ('guitar_strings', 'drum_sticks', 'capo', 'tuner'):
        base *= min(pack * 0.60, 3.0)

    return round(base, 2)


def _estimate_toy_price(title, brand, specs, flags):
    """Estimate toy & game price."""
    t = title.lower()

    if re.search(r'\b(lego|building set|building block|mega bloks|magnetic block|magnet tiles?|magna.?tiles?)\b', t):
        ttype = 'building_set'
    elif re.search(r'\b(action figure|figurine)\b', t):      ttype = 'action_figure'
    elif re.search(r'\b(dollhouse|doll house)\b', t):         ttype = 'dollhouse'
    elif re.search(r'\b(doll|barbie)\b', t):                  ttype = 'doll'
    elif re.search(r'\b(plush|stuffed|squishmallow)\b', t):   ttype = 'plush'
    elif re.search(r'\b(rc car|remote control car|rc truck)\b', t): ttype = 'rc_car'
    elif re.search(r'\b(rc drone|toy drone)\b', t):           ttype = 'rc_drone'
    elif re.search(r'\b(board game|tabletop game)\b', t):     ttype = 'board_game'
    elif re.search(r'\b(card game|trading card)\b', t):       ttype = 'card_game'
    elif re.search(r'\b(puzzle|jigsaw)\b', t):                ttype = 'puzzle'
    elif re.search(r'\b(power wheels|ride on|electric car)\b', t): ttype = 'power_wheels'
    elif re.search(r'\b(playhouse|play house)\b', t):         ttype = 'playhouse'
    elif re.search(r'\b(art supplies?|crayons?|markers?|paint|coloring)\b', t): ttype = 'art_supplies'
    elif re.search(r'\b(science kit|stem|experiment)\b', t):  ttype = 'science_kit'
    elif re.search(r'\b(slime)\b', t):                        ttype = 'slime'
    elif re.search(r'\b(nerf|blaster|dart gun)\b', t):        ttype = 'nerf'
    elif re.search(r'\b(water gun|squirt gun)\b', t):         ttype = 'water_gun'
    elif re.search(r'\b(train set|railway)\b', t):            ttype = 'train_set'
    elif re.search(r'\b(play kitchen|toy kitchen)\b', t):     ttype = 'play_kitchen'
    elif re.search(r'\b(play tent|teepee|pop.?up tent)\b', t): ttype = 'play_tent'
    elif re.search(r'\b(trampoline)\b', t):                   ttype = 'trampoline_toy'
    elif re.search(r'\b(swing)\b', t):                        ttype = 'swing'
    elif re.search(r'\b(robot|coding|programmable)\b', t):    ttype = 'robot_toy'
    elif re.search(r'\b(educational|learning|montessori)\b', t): ttype = 'educational_toy'
    elif re.search(r'\b(baby toy|infant toy|rattle)\b', t):   ttype = 'baby_toy'
    else:
        ttype = 'action_figure'

    base = _TOY_TYPE_PRICES.get(ttype, 20)
    base *= _TOY_BRAND_FACTORS.get(brand, 0.90)

    # Piece count pricing for building sets
    if ttype == 'building_set':
        piece_m = re.search(r'(\d+)\s*piece', t)
        if piece_m:
            pieces = int(piece_m.group(1))
            if 'lego' in t:
                ppp = 0.10   # ~$0.10/piece LEGO standard
            elif 'magna' in t or 'magnet' in t:
                ppp = 0.60   # magnetic tiles are ~$0.50-0.80/piece
            elif 'mega bloks' in t or 'duplo' in t:
                ppp = 0.08
            else:
                ppp = 0.10
            base = max(base, pieces * ppp)

    if flags.get('is_premium', 0): base *= 1.15

    return round(base, 2)


def _estimate_pet_price(title, brand, specs, flags):
    """Estimate pet supply price."""
    t = title.lower()

    if re.search(r'\b(cat tree|cat tower|scratching post|cat condo)\b', t): ptype = 'cat_tree'
    elif re.search(r'\b(dog food|puppy food)\b', t):       ptype = 'dog_food'
    elif re.search(r'\b(cat food|kitten food)\b', t):      ptype = 'cat_food'
    elif re.search(r'\b(dog treat|puppy treat)\b', t):     ptype = 'dog_treat'
    elif re.search(r'\b(cat treat)\b', t):                  ptype = 'cat_treat'
    elif re.search(r'\b(dog bed)\b', t):                    ptype = 'dog_bed'
    elif re.search(r'\b(cat bed)\b', t):                    ptype = 'cat_bed'
    elif re.search(r'\b(dog crate|kennel)\b', t):           ptype = 'dog_crate'
    elif re.search(r'\b(leash)\b', t):                      ptype = 'leash'
    elif re.search(r'\b(collar)\b', t):                     ptype = 'collar'
    elif re.search(r'\b(harness)\b', t):                    ptype = 'harness'
    elif re.search(r'\b(dog toy|chew toy|fetch|frisbee)\b', t): ptype = 'dog_toy'
    elif re.search(r'\b(cat toy|feather|laser)\b', t):      ptype = 'cat_toy'
    elif re.search(r'\b(aquarium|fish tank)\b', t):         ptype = 'aquarium'
    elif re.search(r'\b(bird cage)\b', t):                  ptype = 'bird_cage'
    elif re.search(r'\b(litter box|litter pan)\b', t):      ptype = 'litter_box'
    elif re.search(r'\b(cat litter|kitty litter)\b', t):    ptype = 'cat_litter'
    elif re.search(r'\b(pet carrier|dog carrier|cat carrier)\b', t): ptype = 'pet_carrier'
    elif re.search(r'\b(grooming|brush|deshedding|nail clipper)\b', t): ptype = 'grooming'
    elif re.search(r'\b(flea|tick|flea treatment)\b', t):   ptype = 'flea_treatment'
    elif re.search(r'\b(pet gate|dog gate)\b', t):          ptype = 'pet_gate'
    elif re.search(r'\b(pet fountain|water fountain)\b', t): ptype = 'pet_fountain'
    elif re.search(r'\b(training pad|pee pad|puppy pad)\b', t): ptype = 'training_pad'
    elif re.search(r'\b(pet ramp|dog ramp|dog stairs)\b', t): ptype = 'pet_ramp'
    elif re.search(r'\b(pet stroller|dog stroller)\b', t):  ptype = 'pet_stroller'
    elif re.search(r'\b(seat cover|back seat|car seat)\b', t): ptype = 'pet_carrier'
    else:
        ptype = 'dog_toy'

    base = _PET_TYPE_PRICES.get(ptype, 20)

    # Size-based adjustments (cat tree height, dog crate size)
    screen = specs.get('screen_inch', 0)  # reused for height
    if ptype == 'cat_tree' and screen > 0:
        if screen >= 70: base *= 1.50
        elif screen >= 50: base *= 1.20

    if flags.get('is_premium', 0): base *= 1.20

    pack = specs.get('pack_qty', 1)
    if pack > 1 and ptype in ('dog_toy', 'cat_toy', 'cat_treat', 'dog_treat', 'training_pad'):
        base *= min(pack * 0.50, 3.0)

    return round(base, 2)


def _estimate_office_price(title, brand, specs, flags):
    """Estimate office product price."""
    t = title.lower()

    if re.search(r'\b(printer|all.?in.?one)\b', t):       otype = 'printer'
    elif re.search(r'\b(scanner)\b', t):                    otype = 'scanner'
    elif re.search(r'\b(shredder)\b', t):                   otype = 'shredder'
    elif re.search(r'\b(laminator)\b', t):                  otype = 'laminator'
    elif re.search(r'\b(label maker)\b', t):                otype = 'label_maker'
    elif re.search(r'\b(calculator)\b', t):                 otype = 'calculator'
    elif re.search(r'\b(whiteboard|dry erase)\b', t):       otype = 'whiteboard'
    elif re.search(r'\b(filing cabinet)\b', t):             otype = 'filing_cabinet'
    elif re.search(r'\b(desk organizer|desk caddy)\b', t):  otype = 'desk_organizer'
    elif re.search(r'\b(monitor stand|laptop stand)\b', t): otype = 'monitor_stand'
    elif re.search(r'\b(ink cartridge|ink)\b', t):          otype = 'ink_cartridge'
    elif re.search(r'\b(toner)\b', t):                      otype = 'toner'
    elif re.search(r'\b(paper|copy paper|cardstock)\b', t): otype = 'paper'
    elif re.search(r'\b(pen|ballpoint|gel pen|rollerball)\b', t): otype = 'pens'
    elif re.search(r'\b(pencil)\b', t):                     otype = 'pencils'
    elif re.search(r'\b(marker|sharpie|dry erase marker)\b', t): otype = 'markers'
    elif re.search(r'\b(highlighter)\b', t):                otype = 'highlighters'
    elif re.search(r'\b(notebook|notepad|legal pad)\b', t): otype = 'notebooks'
    elif re.search(r'\b(planner|calendar|agenda)\b', t):    otype = 'planner'
    elif re.search(r'\b(sticky note|post.?it)\b', t):       otype = 'sticky_notes'
    elif re.search(r'\b(tape|packing tape|scotch)\b', t):   otype = 'tape'
    elif re.search(r'\b(glue|adhesive|glue stick)\b', t):   otype = 'glue'
    elif re.search(r'\b(stapler)\b', t):                    otype = 'stapler'
    elif re.search(r'\b(scissors)\b', t):                   otype = 'scissors'
    elif re.search(r'\b(envelope)\b', t):                   otype = 'envelope'
    elif re.search(r'\b(binder)\b', t):                     otype = 'binder'
    elif re.search(r'\b(folder|file folder|expanding file)\b', t): otype = 'folder'
    elif re.search(r'\b(webcam)\b', t):                     otype = 'webcam'
    elif re.search(r'\b(headset)\b', t):                    otype = 'headset_office'
    elif re.search(r'\b(usb hub|hub)\b', t):                otype = 'usb_hub'
    elif re.search(r'\b(docking station|dock)\b', t):       otype = 'docking_station'
    elif re.search(r'\b(mouse pad|desk pad|desk mat)\b', t): otype = 'desk_pad'
    else:
        otype = 'pens'

    base = _OFFICE_TYPE_PRICES.get(otype, 12)

    office_brands = {'brother': 1.10, 'hp': 1.05, 'epson': 1.05,
                     'canon': 1.05, 'fellowes': 1.10, 'logitech': 1.10}
    base *= office_brands.get(brand, 0.85)

    pack = specs.get('pack_qty', 1)
    if pack > 1 and otype in ('pens', 'pencils', 'markers', 'highlighters',
                               'notebooks', 'folders', 'binder', 'envelope',
                               'sticky_notes', 'glue', 'tape'):
        # Cheap consumables: 10-pack of pens ~ $5-8, not 2.5x single price
        if otype in ('pens', 'pencils', 'glue', 'tape'):
            base *= max(min(pack * 0.12, 1.5), 1.0)  # At least 1.0×, max 1.5×
        else:
            base *= min(pack * 0.25, 2.0)

    return round(base, 2)


def _estimate_book_price(title, brand, specs, flags):
    """Estimate book & media price."""
    t = title.lower()

    if re.search(r'\b(textbook|college|university|edition)\b', t): btype = 'textbook'
    elif re.search(r'\b(art book|photography book|coffee table book)\b', t): btype = 'art_book'
    elif re.search(r'\b(cookbook|recipe)\b', t):             btype = 'cookbook'
    elif re.search(r'\b(children|kids|baby book|board book|picture book)\b', t): btype = 'childrens_book'
    elif re.search(r'\b(comic|graphic novel)\b', t):        btype = 'comic'
    elif re.search(r'\b(manga)\b', t):                      btype = 'manga'
    elif re.search(r'\b(vinyl|record|lp)\b', t):            btype = 'vinyl_record'
    elif re.search(r'\b(dvd|blu-?ray)\b', t):
        btype = 'blu_ray' if 'blu' in t else 'dvd'
    elif re.search(r'\b(cd|album)\b', t):                   btype = 'cd'
    elif re.search(r'\b(calendar)\b', t):                   btype = 'calendar'
    elif re.search(r'\b(journal|diary)\b', t):              btype = 'journal'
    elif re.search(r'\b(planner)\b', t):                    btype = 'calendar'
    elif re.search(r'\b(coloring book)\b', t):              btype = 'coloring_book'
    elif re.search(r'\b(sticker)\b', t):                    btype = 'sticker_book'
    elif re.search(r'\b(art supplies?|markers?|paint|sketch)\b', t): btype = 'art_supplies_media'
    elif re.search(r'\b(craft|sewing|knitting|crochet)\b', t): btype = 'craft_supplies'
    elif re.search(r'\b(hardcover|hard cover)\b', t):       btype = 'hardcover'
    elif re.search(r'\b(paperback|soft cover)\b', t):       btype = 'paperback'
    else:
        btype = 'paperback'

    base = _BOOK_TYPE_PRICES.get(btype, 15)

    pack = specs.get('pack_qty', 1)
    if pack > 1: base *= min(pack * 0.55, 3.0)

    return round(base, 2)


def _estimate_health_price(title, brand, specs, flags):
    """Estimate health & household price."""
    t = title.lower()

    if re.search(r'\b(vitamin|multivitamin)\b', t):        htype = 'vitamin'
    elif re.search(r'\b(supplement|probiotic|omega|fish oil|collagen|melatonin)\b', t): htype = 'supplement'
    elif re.search(r'\b(protein powder|whey|creatine|pre.?workout|bcaa)\b', t): htype = 'protein_powder'
    elif re.search(r'\b(first aid|bandage|band-?aid)\b', t): htype = 'first_aid'
    elif re.search(r'\b(thermometer)\b', t):                htype = 'thermometer'
    elif re.search(r'\b(blood pressure|bp monitor)\b', t):  htype = 'blood_pressure'
    elif re.search(r'\b(pulse oximeter|oximeter)\b', t):    htype = 'pulse_oximeter'
    elif re.search(r'\b(nebulizer)\b', t):                  htype = 'nebulizer'
    elif re.search(r'\b(hearing aid)\b', t):                htype = 'hearing_aid'
    elif re.search(r'\b(cpap)\b', t):                       htype = 'cpap'
    elif re.search(r'\b(wheelchair)\b', t):                 htype = 'wheelchair'
    elif re.search(r'\b(walker|rollator)\b', t):            htype = 'walker_medical'
    elif re.search(r'\b(knee brace|ankle brace|wrist brace|elbow brace)\b', t): htype = 'knee_brace'
    elif re.search(r'\b(back brace|posture corrector)\b', t): htype = 'back_brace'
    elif re.search(r'\b(compression|compression sock)\b', t): htype = 'compression'
    elif re.search(r'\b(essential oil|aromatherapy)\b', t): htype = 'essential_oil'
    elif re.search(r'\b(diffuser)\b', t):                   htype = 'diffuser'
    elif re.search(r'\b(humidifier)\b', t):                 htype = 'humidifier_health'
    elif re.search(r'\b(air purifier)\b', t):               htype = 'air_purifier_health'
    elif re.search(r'\b(water filter|brita|pur)\b', t):     htype = 'water_filter'
    elif re.search(r'\b(cleaning|cleaner|disinfectant|lysol|clorox)\b', t): htype = 'cleaning_supplies'
    elif re.search(r'\b(detergent|laundry)\b', t):          htype = 'detergent'
    elif re.search(r'\b(trash bags?|garbage bags?)\b', t):  htype = 'trash_bags'
    elif re.search(r'\b(paper towels?)\b', t):              htype = 'paper_towels'
    elif re.search(r'\b(toilet paper|tissues?)\b', t):      htype = 'toilet_paper'
    elif re.search(r'\b(hand soap|dish soap)\b', t):        htype = 'hand_soap'
    elif re.search(r'\b(hand sanitizer)\b', t):             htype = 'hand_sanitizer'
    elif re.search(r'\b(mask|face mask|disposable mask)\b', t) and not re.search(r'\b(skin|beauty|sheet)\b', t):
        htype = 'mask'
    elif re.search(r'\b(glove|latex|nitrile)\b', t):        htype = 'gloves_medical'
    elif re.search(r'\b(battery|batteries|duracell|energizer|alkaline)\b', t): htype = 'battery'
    elif re.search(r'\b(sandwich bag|storage bag|freezer bag|ziploc|ziplock|zip.?loc)\b', t): htype = 'plastic_bags'
    elif re.search(r'\b(aluminum foil|plastic wrap|cling wrap|parchment|wax paper)\b', t): htype = 'kitchen_wrap'
    elif re.search(r'\b(sponge|scrub pad|steel wool|dish cloth)\b', t): htype = 'sponges'
    elif re.search(r'\b(air freshener|febreze|glade)\b', t): htype = 'air_freshener'
    elif re.search(r'\b(roller skate|inline skate|ice skate)\b', t):
        return round(60, 2)  # misclassified sports item
    else:
        htype = 'vitamin'

    base = _HEALTH_TYPE_PRICES.get(htype, 18)

    health_brands = {'levoit': 1.20, 'honeywell': 1.05, 'garden of life': 1.20,
                     'nature made': 0.90, 'optimum nutrition': 1.10}
    base *= health_brands.get(brand, 0.90)

    pack = specs.get('pack_qty', 1)
    # For consumables like trash bags, paper towels, "80 count" means 80 individual bags
    # in a SINGLE box — NOT 80 boxes. Only apply multiplier to pack/box types.
    if pack > 1 and htype in ('vitamin', 'supplement', 'mask', 'gloves_medical'):
        base *= min(pack * 0.50, 3.0)
    elif pack > 1 and htype in ('battery', 'plastic_bags'):
        # Batteries: 20-pack ~ $12-18, bags: 100-count ~ $5-8
        base *= min(pack * 0.10, 2.0)
    # Don't multiply for trash_bags, paper_towels, toilet_paper — "80 count" is inside ONE box

    return round(base, 2)


# ================================================================
# PRODUCT TYPE DETECTION — COMPREHENSIVE
# ================================================================
def _detect_product_type(title, category, specs):
    """
    Detect the high-level product type from title + category + specs.
    Returns one of 24+ product types or None.
    """
    t = title.lower()
    cat_lower = category.lower()

    # ── First: use CATEGORY to route ──
    # If category is unambiguous (e.g., "Musical Instruments"), trust it.

    # Electronics - Computers → laptop (or e-reader/3d printer)
    if 'computer' in cat_lower:
        # E-readers (often miscategorized as computers)
        if re.search(r'\b(kindle|e-?reader|kobo|nook)\b', t):
            return 'smart_home'  # Route to smart_home for generic electronics
        # 3D printers
        if re.search(r'\b(3d printer|resin printer|fdm printer|elegoo|creality|anycubic|prusa)\b', t):
            return None  # Use reference price / ML only
        laptop_kw = ['laptop', 'notebook', 'chromebook', 'ultrabook']
        cpu_kw = ['core i', 'i3', 'i5', 'i7', 'i9', 'ryzen', 'celeron',
                  'pentium', 'athlon', 'ram', 'ssd', 'hdd']
        is_laptop_explicit = any(kw in t for kw in laptop_kw)
        cpu_hints = sum(1 for kw in cpu_kw if kw in t)
        if is_laptop_explicit or cpu_hints >= 2:
            return 'laptop'
        # Even if not laptop keywords, if it's in Computers category, try laptop heuristic
        if cpu_hints >= 1:
            return 'laptop'

    # Electronics - Mobile → phone (only if actual phone, not accessory)
    if 'mobile' in cat_lower:
        phone_kw = ['smartphone', 'cell phone', 'mobile phone']
        phone_brands = ['iphone', 'galaxy s2', 'galaxy s3', 'galaxy s4',
                        'galaxy s5', 'galaxy s6', 'galaxy s7', 'galaxy s8',
                        'galaxy s9', 'galaxy s10', 'galaxy s20', 'galaxy s21',
                        'galaxy s22', 'galaxy s23', 'galaxy s24', 'galaxy s25',
                        'galaxy a', 'galaxy z', 'galaxy m', 'galaxy f',
                        'pixel', 'oneplus', 'nord', 'redmi', 'poco',
                        'moto g', 'moto e', 'moto razr', 'nothing phone',
                        'infinix', 'tecno', 'itel', 'realme', 'vivo',
                        'oppo', 'honor', 'huawei', 'xiaomi', 'nokia',
                        'motorola', 'zte', 'alcatel', 'blu', 'umidigi',
                        'lava', 'micromax', 'tcl']
        acc_kw = ['case', 'cover', 'cable', 'charger', 'adapter', 'screen protector',
                  'holder', 'phone stand', 'mount', 'tripod', 'ring light', 'selfie']
        if any(kw in t for kw in acc_kw):
            return None  # It's an accessory, not a phone
        if any(kw in t for kw in phone_kw) or any(kw in t for kw in phone_brands):
            return 'phone'
        if specs.get('storage_gb', 0) >= 32 or specs.get('ram_gb', 0) >= 2:
            return 'phone'
        # Fallback: category says Mobile and it's not an accessory → likely a phone
        return 'phone'

    # Electronics - TV & Video
    if 'tv' in cat_lower or 'video' in cat_lower:
        tv_kw = ['television', ' tv ', 'smart tv', 'led tv', 'oled tv', 'qled',
                 'neo qled', 'mini led']
        if any(kw in t for kw in tv_kw) or specs.get('screen_inch', 0) >= 24:
            return 'tv'
        return 'tv'  # Category says TV, trust it

    # Electronics - Audio
    if 'audio' in cat_lower:
        return 'audio'

    # Electronics - Cameras
    if 'camera' in cat_lower:
        return 'camera'

    # Electronics - Gaming
    if 'gaming' in cat_lower:
        return 'gaming'

    # Electronics - Wearables
    if 'wearable' in cat_lower:
        return 'wearable'

    # Electronics - Smart Home
    if 'smart home' in cat_lower:
        return 'smart_home'

    # Clothing & Accessories (but NOT 'Mobile & Accessories' or 'Electronics')
    if 'clothing' in cat_lower or \
       ('accessories' in cat_lower and 'mobile' not in cat_lower and 'electronic' not in cat_lower):
        return 'clothing'

    # Home & Kitchen - Appliances
    if 'appliance' in cat_lower:
        return 'appliance'

    # Home & Kitchen - Furniture
    if 'furniture' in cat_lower:
        return 'furniture'

    # Home & Kitchen - Decor
    if 'decor' in cat_lower:
        return 'decor'

    # Tools & Home Improvement
    if 'tools' in cat_lower or 'improvement' in cat_lower:
        return 'tools'

    # Garden & Outdoor
    if 'garden' in cat_lower or ('outdoor' in cat_lower and 'sport' not in cat_lower):
        return 'garden'

    # Sports & Outdoors
    if 'sport' in cat_lower:
        return 'sports'

    # Automotive
    if 'automotive' in cat_lower or 'auto' in cat_lower:
        return 'automotive'

    # Baby Products
    if 'baby' in cat_lower:
        return 'baby'

    # Beauty & Personal Care
    if 'beauty' in cat_lower or 'personal care' in cat_lower:
        return 'beauty'

    # Musical Instruments
    if 'musical' in cat_lower or 'instrument' in cat_lower:
        return 'music'

    # Toys & Games
    if 'toys' in cat_lower or 'games' in cat_lower:
        return 'toys'

    # Pet Supplies
    if 'pet' in cat_lower:
        return 'pet'

    # Office Products
    if 'office' in cat_lower:
        return 'office'

    # Books & Media
    if 'book' in cat_lower or 'media' in cat_lower:
        return 'books'

    # Health & Household
    if 'health' in cat_lower or 'household' in cat_lower:
        return 'health'

    # ── Fallback: try keyword detection regardless of category ──
    # Phone detection
    phone_kw = ['smartphone', 'cell phone', 'mobile phone']
    phone_brands_fb = ['iphone', 'galaxy s2', 'galaxy s23', 'galaxy s24',
                       'galaxy s25', 'galaxy z', 'pixel']
    if any(kw in t for kw in phone_kw) or any(kw in t for kw in phone_brands_fb):
        return 'phone'

    # Laptop detection
    if re.search(r'\b(laptop|notebook|chromebook)\b', t):
        return 'laptop'

    # Tablet detection
    if re.search(r'\b(tablet|ipad|galaxy tab)\b', t):
        return 'tablet'

    return None


def _apply_smart_heuristic(ml_price, title, category, brand, specs, flags):
    """
    Apply category-aware smart heuristic estimation.
    Routes to the appropriate estimator based on detected product type.
    Blends with ML price based on how far off the ML appears to be.
    """
    product_type = _detect_product_type(title, category, specs)

    if product_type is None:
        return ml_price, None

    # ── Route to the right estimator ──
    estimators = {
        'laptop':     lambda: _estimate_laptop_price(title, brand, specs, flags),
        'phone':      lambda: _estimate_phone_price(title, brand, specs, flags),
        'tv':         lambda: _estimate_tv_price(title, brand, specs, flags),
        'tablet':     lambda: _estimate_tablet_price(title, brand, specs, flags),
        'audio':      lambda: _estimate_audio_price(title, brand, specs, flags),
        'camera':     lambda: _estimate_camera_price(title, brand, specs, flags),
        'gaming':     lambda: _estimate_gaming_price(title, brand, specs, flags),
        'wearable':   lambda: _estimate_wearable_price(title, brand, specs, flags),
        'smart_home': lambda: _estimate_smarthome_price(title, brand, specs, flags),
        'clothing':   lambda: _estimate_clothing_price(title, brand, specs, flags),
        'appliance':  lambda: _estimate_appliance_price(title, brand, specs, flags),
        'furniture':  lambda: _estimate_furniture_price(title, brand, specs, flags),
        'decor':      lambda: _estimate_decor_price(title, brand, specs, flags),
        'tools':      lambda: _estimate_tools_price(title, brand, specs, flags),
        'garden':     lambda: _estimate_garden_price(title, brand, specs, flags),
        'sports':     lambda: _estimate_sports_price(title, brand, specs, flags),
        'automotive': lambda: _estimate_auto_price(title, brand, specs, flags),
        'baby':       lambda: _estimate_baby_price(title, brand, specs, flags),
        'beauty':     lambda: _estimate_beauty_price(title, brand, specs, flags),
        'music':      lambda: _estimate_music_price(title, brand, specs, flags),
        'toys':       lambda: _estimate_toy_price(title, brand, specs, flags),
        'pet':        lambda: _estimate_pet_price(title, brand, specs, flags),
        'office':     lambda: _estimate_office_price(title, brand, specs, flags),
        'books':      lambda: _estimate_book_price(title, brand, specs, flags),
        'health':     lambda: _estimate_health_price(title, brand, specs, flags),
    }

    estimator = estimators.get(product_type)
    if estimator is None:
        return ml_price, None

    heuristic = estimator()

    # ── Blending logic ──
    # Ratio of ML prediction to heuristic estimate
    ratio = ml_price / heuristic if heuristic > 0 else 1.0

    # Ultra-luxury: heuristic is very high (>$2000), ML is way below → trust heuristic 90%
    if heuristic >= 2000 and ratio < 0.15:
        final = heuristic * 0.90 + ml_price * 0.10
    # Cheap accessories/consumables: heuristic < $60 and ML is way too high
    elif heuristic < 60 and ratio > 3.0:
        final = heuristic * 0.92 + ml_price * 0.08
    elif heuristic < 60 and ratio > 2.0:
        final = heuristic * 0.80 + ml_price * 0.20
    elif ratio < 0.40:
        # ML is way off — heavily favor heuristic
        final = heuristic * 0.75 + ml_price * 0.25
    elif ratio < 0.70:
        # ML is somewhat low — balanced blend
        final = heuristic * 0.50 + ml_price * 0.50
    elif ratio > 2.50:
        # ML is way too high — favor heuristic
        final = heuristic * 0.65 + ml_price * 0.35
    elif ratio > 1.60:
        # ML is somewhat high — balanced blend
        final = heuristic * 0.40 + ml_price * 0.60
    else:
        # ML is in the right ballpark — trust ML
        final = ml_price

    return round(final, 2), product_type


def _estimate_tablet_price(title, brand, specs, flags):
    """Estimate tablet price."""
    t = title.lower()
    base = 350
    if brand == 'apple':       base = 500
    elif brand == 'samsung':   base = 400
    elif brand == 'microsoft': base = 600
    elif brand == 'amazon':    base = 120
    elif brand == 'lenovo':    base = 250

    if specs.get('storage_gb', 0) >= 512: base *= 1.30
    elif specs.get('storage_gb', 0) >= 256: base *= 1.15
    if re.search(r'\b(pro)\b', t): base *= 1.50
    if re.search(r'\b(air)\b', t) and brand == 'apple': base *= 1.10
    if re.search(r'\b(mini)\b', t): base *= 0.80
    if flags.get('is_budget', 0): base *= 0.55
    if re.search(r'\b(refurbished|renewed)\b', t): base *= 0.55

    return round(base, 2)


# ================================================================
# PREDICTION FUNCTION
# ================================================================
def predict_price(title, category, subcategory=None):
    """
    Predict the price for a product based on title and category.

    Uses a two-stage approach:
      1. ML model prediction (GradientBoosting on 50k+ products)
      2. Reference price calibration for known products

    Args:
        title: Product title (str)
        category: Product category (str)
        subcategory: Product subcategory (str, optional)

    Returns:
        dict with predicted price, price range, and confidence info
    """
    if subcategory is None:
        subcategory = category

    # 1. TF-IDF features
    X_tfidf = tfidf.transform([title])

    # 2. Category encoding
    cat_enc = safe_label_encode(le_cat, category)
    subcat_enc = safe_label_encode(le_subcat, subcategory)

    # 3. Brand
    brand = extract_brand(title)
    brand_enc = safe_label_encode(le_brand, brand)

    # 4. Specs
    specs = extract_specs(title)

    # 5. Quality flags
    flags = extract_quality_flags(title)

    # 6. Title stats
    title_length = len(title)
    title_word_count = len(title.split())

    # 7. Accessory & device detection
    accessory_kw = ['case', 'cover', 'protector', 'cable', 'charger', 'adapter',
                    'holder', 'phone stand', 'laptop stand', 'tablet stand',
                    'strap', 'wrist band', 'watch band', 'sleeve',
                    'replacement', 'refill', 'film', 'tempered glass',
                    'screen protector', 'cleaning cloth', 'lens cap',
                    'stylus', 'pen tip', 'nib', 'ear tip', 'ear pad',
                    'remote control', 'battery pack', 'carrying case',
                    'tripod', 'selfie stick', 'ring light',
                    'memory card', 'sd card', 'usb hub', 'dongle',
                    'hdmi cable', 'extension cord', 'power strip',
                    'wall plate', 'wall mount', 'bracket']
    is_accessory = int(any(kw in title.lower() for kw in accessory_kw))

    device_kw = ['phone', 'smartphone', 'laptop', 'notebook', 'computer', 'tablet',
                 'ipad', 'macbook', 'television', 'monitor', 'camera', 'drone',
                 'vacuum', 'washer', 'dryer', 'dishwasher', 'refrigerator',
                 'printer', 'scanner', 'projector', 'speaker', 'soundbar',
                 'headphones', 'earbuds', 'console', 'playstation', 'xbox',
                 'nintendo', 'guitar', 'piano', 'drum', 'treadmill', 'bicycle',
                 'scooter', 'robot', 'mixer', 'blender', 'oven', 'microwave',
                 'air purifier', 'humidifier', 'air conditioner']
    is_main_device = int(any(kw in title.lower() for kw in device_kw))

    # 8. Category/brand median features (must match training)
    cat_med = cat_median_log_price.get(category, global_median_log_price)
    subcat_med = subcat_median_log_price.get(subcategory, cat_med)
    brand_med = brand_median_log_price.get(brand, global_median_log_price)
    cat_mn = cat_mean_log_price.get(category, global_median_log_price)
    brand_mn = brand_mean_log_price.get(brand, global_median_log_price)

    # Build feature vector — ORDER MUST MATCH retrain_model.py exactly!
    numeric_values = [
        specs['storage_gb'], specs['storage_tb'], specs['screen_inch'],
        specs['battery_mah'], specs['wattage'], specs['pack_qty'],
        flags['is_premium'], flags['is_budget'], flags['is_wireless'],
        flags['is_waterproof'], flags['is_organic'], flags['has_led'],
        flags['is_smart'],
        is_accessory, is_main_device,
        title_length, title_word_count,
        cat_med, subcat_med, brand_med, cat_mn, brand_mn,
    ]

    X_cat = csr_matrix(np.array([[cat_enc, subcat_enc, brand_enc]]))
    X_numeric = csr_matrix(np.array([numeric_values]))
    X = hstack([X_tfidf, X_cat, X_numeric])

    # Predict log price
    log_price_pred = model.predict(X)[0]
    ml_raw_price = np.expm1(log_price_pred)

    # ── Reference Price Calibration ──────────────────────────────
    ref_price, ref_key = find_reference_price(title)
    reference_used = None

    # Guard: if the reference match is for a main device (console, camera body, etc.)
    # but the product is actually an accessory, skip the device reference.
    if ref_price is not None and ref_key:
        _acc_guard_kw = [
            # Gaming accessories
            'controller', 'gamepad', 'joystick', 'remote', 'media remote',
            'cable', 'charger', 'case', 'cover', 'adapter', 'skin',
            'dualsense', 'dualshock', 'joy-con', 'joycon', 'pro controller',
            'headset', 'charging station', 'charging dock', 'dock set',
            'vertical stand', 'cover plate', 'faceplate',
            'screen protector', 'tempered glass', 'thumb grip',
            'game pass', 'subscription', 'gift card',
            # Multi-language controller terms
            'manette', 'mando', 'controlador', 'kontroller', 'telecomando',
            # Camera accessories
            'battery', 'lens cap', 'lens hood', 'housing',
            'protective housing', 'propeller', 'propeller guard',
            'nd filter', 'uv filter', 'camera bag', 'camera strap',
            'memory card', 'sd card',
            # Phone/device accessories
            'clear case', 'phone case', 'silicone case', 'wallet case',
            'screen protector', 'tempered glass',
            'ear tip', 'ear pad', 'replacement tip', 'cushion',
            'band', 'watch band', 'wrist strap',
            'keyboard', 'mouse', 'stylus', 'pen tip',
        ]
        # Game titles (contain console name but are games, not consoles)
        _game_kw = ['game', 'edition', 'legacy', 'hogwarts', 'starfield',
                    'mario kart', 'zelda', 'god of war', 'spider-man',
                    'final fantasy', 'call of duty', 'fifa', 'fortnite',
                    'elden ring', 'ratchet', 'gran turismo', 'deluxe']

        _device_refs = ['ps5', 'playstation', 'xbox', 'nintendo switch',
                        'steam deck',
                        'sony a7', 'sony a9', 'sony a6', 'canon eos',
                        'canon r', 'nikon z', 'gopro hero', 'dji mini',
                        'dji mavic', 'dji air',
                        # Phone models (case/cover/cable matching phone ref)
                        'iphone', 'galaxy s2', 'galaxy s23', 'galaxy s24',
                        'galaxy s25', 'galaxy z', 'galaxy a',
                        'pixel 9', 'pixel 8', 'pixel 7',
                        'macbook', 'surface pro', 'surface laptop',
                        'ipad pro', 'ipad air', 'ipad mini', 'ipad 10',
                        'galaxy tab',
                        # Laptops
                        'thinkpad', 'xps 1', 'hp pavilion',
                        # Audio devices (ear tips matching airpods ref)
                        'airpods', 'bose quietcomfort', 'sony wh-',
                        'sony wf-', 'beats studio', 'beats solo',
                        ]

        t_low = title.lower()
        is_acc_title = any(ag in t_low for ag in _acc_guard_kw)
        is_game_title = any(gk in t_low for gk in _game_kw)
        matches_device_ref = any(cr in ref_key for cr in _device_refs)

        # If the ref key ITSELF describes an accessory, it's fine — keep it.
        _acc_ref_markers = ['controller', 'headset', 'charging', 'remote',
                            'camera', 'cover', 'pulse 3d', 'media remote',
                            'game pass', 'dock', 'stand', 'case', 'ear tip',
                            'battery', 'batteries']
        ref_is_acc = any(am in ref_key for am in _acc_ref_markers)

        # Also treat 'dualsense' ref as a device ref when the product is NOT a controller
        _controller_names = ['dualsense', 'dualshock', 'joy-con']
        is_controller_ref = any(cn in ref_key for cn in _controller_names)
        title_is_not_controller = not re.search(
            r'\b(controller|gamepad|joystick|manette|mando)\b', title.lower())
        product_is_station = any(kw in title.lower() for kw in [
            'charging station', 'charging dock', 'dock set', 'stand',
            'cover plate', 'faceplate', 'skin', 'decal'])

        if is_controller_ref and title_is_not_controller and product_is_station:
            ref_price, ref_key = None, None  # Don't use controller ref for its dock
        elif (is_acc_title or is_game_title) and matches_device_ref and not ref_is_acc:
            # Device ref matched but product is an accessory/game.
            # Try again for a more specific ref (e.g. "dualsense")
            ref_price2, ref_key2 = _find_reference_excluding(title, _device_refs)
            if ref_price2 is not None:
                ref_price, ref_key = ref_price2, ref_key2
            else:
                ref_price, ref_key = None, None

    if ref_price is not None and not is_accessory:
        # Blend: reference price anchors the prediction for known products.
        # Apply spec-based adjustments on top of the reference base.
        ref_adjusted = _apply_spec_adjustments(ref_price, specs, flags)

        # Weighted blend: 85% reference, 15% ML
        # Reference prices are market-verified, so they should dominate.
        price_pred = ref_adjusted * 0.85 + ml_raw_price * 0.15
        reference_used = ref_key
    else:
        # No reference match — apply smart heuristic for known product types
        # but NOT for accessories (cables, cases, etc.)
        if not is_accessory:
            price_pred, product_type = _apply_smart_heuristic(
                ml_raw_price, title, category, brand, specs, flags
            )
            if product_type:
                reference_used = f"heuristic:{product_type}"
            else:
                price_pred = ml_raw_price
        else:
            price_pred = ml_raw_price

    # Accessory price cap: accessories are typically $5–$50
    # Apply type-specific caps for known cheap accessories
    if is_accessory:
        t_low = title.lower()
        if any(kw in t_low for kw in ['case', 'cover', 'skin', 'sleeve', 'pouch',
                                       'clear case', 'silicone', 'wallet case']):
            price_pred = min(price_pred, 35.00)
        elif any(kw in t_low for kw in ['ear tip', 'replacement tip', 'nib', 'pen tip']):
            price_pred = min(price_pred, 15.00)
        elif any(kw in t_low for kw in ['ear pad', 'ear cushion']):
            # AirPods Max cushions are $69, don't over-cap
            price_pred = min(price_pred, 80.00)
        elif any(kw in t_low for kw in ['screen protector', 'tempered glass', 'film']):
            price_pred = min(price_pred, 20.00)
        elif any(kw in t_low for kw in ['cable', 'cord', 'usb', 'lightning', 'hdmi cable']):
            price_pred = min(price_pred, 30.00)
        elif price_pred > 80:
            price_pred = min(price_pred, 49.99)

    # Price range (± margin based on model error)
    pct_error = metadata.get('median_pct_error', 20)
    margin = price_pred * (pct_error / 100)
    price_low = max(0.99, price_pred - margin)
    price_high = price_pred + margin

    return {
        'predicted_price': round(price_pred, 2),
        'price_range': (round(price_low, 2), round(price_high, 2)),
        'brand_detected': brand,
        'specs': specs,
        'flags': flags,
        'reference_used': reference_used,
        'ml_raw_price': round(ml_raw_price, 2),
    }


# ================================================================
# DISPLAY FUNCTIONS
# ================================================================
def display_categories():
    """Display available categories."""
    print("\n📂 Available categories:")
    for i, cat in enumerate(categories, 1):
        print(f"   {i:>2}. {cat}")
    return categories


def display_result(result, title):
    """Display prediction result."""
    print(f"\n{'=' * 60}")
    print(f"💰 PRICE PREDICTION RESULT")
    print(f"{'=' * 60}")
    print(f"\n   📦 Product:  {title[:80]}{'...' if len(title) > 80 else ''}")
    print(f"   🏷️  Brand:    {result['brand_detected']}")
    print(f"\n   ┌────────────────────────────────────────┐")
    print(f"   │  💵 Suggested Price: ${result['predicted_price']:.2f}")
    print(f"   │  📊 Price Range:     ${result['price_range'][0]:.2f} — ${result['price_range'][1]:.2f}")
    print(f"   └────────────────────────────────────────┘")

    # Show detected specs
    specs = result['specs']
    active_specs = {k: v for k, v in specs.items()
                    if not k.startswith('_') and k != 'pack_qty'
                    and isinstance(v, (int, float)) and v > 0}
    if active_specs or specs['pack_qty'] > 1:
        print(f"\n   🔧 Detected specs:")
        for k, v in active_specs.items():
            label = k.replace('_', ' ').title()
            print(f"      • {label}: {v}")
        if specs['pack_qty'] > 1:
            print(f"      • Pack Quantity: {specs['pack_qty']}")

    # Show quality flags
    flags = result['flags']
    active_flags = [k.replace('_', ' ').replace('is ', '').replace('has ', '').title()
                    for k, v in flags.items() if v == 1]
    if active_flags:
        print(f"\n   ✨ Detected attributes: {', '.join(active_flags)}")

    print(f"\n{'=' * 60}")


# ================================================================
# INTERACTIVE MODE
# ================================================================
def main():
    """Main interactive loop for sellers."""
    while True:
        print(f"\n{'─' * 60}")
        print("🆕 NEW PRODUCT LISTING")
        print(f"{'─' * 60}")

        # Get title
        title = input("\n📝 Enter product title (or 'quit' to exit):\n   → ").strip()
        if title.lower() in ('quit', 'exit', 'q'):
            print("\n👋 Goodbye! Happy selling!")
            break
        if not title:
            print("   ⚠️ Title cannot be empty!")
            continue

        # Show and select category
        cats = display_categories()
        cat_input = input(f"\n📂 Choose category (1-{len(cats)}) or type name:\n   → ").strip()

        if cat_input.isdigit() and 1 <= int(cat_input) <= len(cats):
            category = cats[int(cat_input) - 1]
        elif cat_input in cats:
            category = cat_input
        else:
            # Fuzzy match
            matches = [c for c in cats if cat_input.lower() in c.lower()]
            if matches:
                category = matches[0]
                print(f"   → Matched: {category}")
            else:
                print(f"   ⚠️ Category not found, using first match or default.")
                category = cats[0]

        print(f"\n   ✅ Category: {category}")

        # Subcategory (optional)
        subcat_input = input("\n📁 Enter subcategory (or press Enter to use category):\n   → ").strip()
        subcategory = subcat_input if subcat_input else category

        # Predict
        print(f"\n   🔮 Predicting price...")
        result = predict_price(title, category, subcategory)
        display_result(result, title)

        # Offer comparison
        another = input("\n🔄 Predict another product? (y/n): ").strip().lower()
        if another not in ('y', 'yes', 'o', 'oui'):
            print("\n👋 Goodbye! Happy selling!")
            break


# ================================================================
# QUICK TEST EXAMPLES
# ================================================================
def run_examples():
    """Run some example predictions to demonstrate the model."""
    print(f"\n{'=' * 60}")
    print("🧪 EXAMPLE PREDICTIONS")
    print(f"{'=' * 60}")

    examples = [
        {
            'title': 'Apple AirPods Pro 2nd Generation with USB-C, Active Noise Cancellation, Wireless Bluetooth Earbuds',
            'category': 'Electronics - Audio',
        },
        {
            'title': 'LEGO Star Wars Millennium Falcon 75375 Building Set, 921 Pieces, Ages 8+',
            'category': 'Toys & Games',
        },
        {
            'title': 'Dyson V15 Detect Absolute Cordless Vacuum Cleaner, Laser Dust Detection, LCD Screen',
            'category': 'Home & Kitchen - Appliances',
        },
        {
            'title': 'Nike Air Max 270 Men Running Shoes, Black/White, Breathable, Size 10',
            'category': 'Clothing & Accessories',
        },
        {
            'title': 'Anker USB-C Charger 20W Fast Charging Wall Adapter for iPhone 15, Samsung Galaxy',
            'category': 'Electronics - Mobile & Accessories',
        },
        {
            'title': 'Organic Green Tea Matcha Powder, 100% Pure Japanese Ceremonial Grade, 4oz',
            'category': 'Grocery & Gourmet Food',
        },
    ]

    for ex in examples:
        result = predict_price(ex['title'], ex['category'])
        print(f"\n   📦 {ex['title'][:70]}...")
        print(f"   📂 {ex['category']}")
        print(f"   🏷️  Brand: {result['brand_detected']}")
        print(f"   💰 Predicted: ${result['predicted_price']:.2f} "
              f"(range: ${result['price_range'][0]:.2f} — ${result['price_range'][1]:.2f})")
        print(f"   {'─' * 50}")


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--examples':
        run_examples()
    else:
        # Show examples first, then interactive mode
        run_examples()
        print(f"\n{'=' * 60}")
        print("🎯 Now try with YOUR products!")
        print(f"{'=' * 60}")
        main()

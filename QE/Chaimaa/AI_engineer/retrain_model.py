"""
Retrain PricePulse model v3:
  1. Accessory vs main product detection
  2. Category/brand median prices as features
  3. Reference price table for known devices
"""

import os
import pandas as pd
import numpy as np
import re
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.sparse import hstack, csr_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 70)
print("RETRAINING PricePulse v3")
print("=" * 70)

# 1. LOAD & CLEAN
df_raw = pd.read_csv(os.path.join(DATA_DIR, 'final_dataset_cleaned.csv'))
df_raw['price_numeric'] = pd.to_numeric(df_raw['price'], errors='coerce')
df_s = df_raw[(df_raw['price_numeric'] > 0) & (df_raw['price_numeric'].notna())].copy()
p995 = df_s['price_numeric'].quantile(0.995)
p005 = df_s['price_numeric'].quantile(0.005)
df_s = df_s[(df_s['price_numeric'] <= p995) & (df_s['price_numeric'] >= p005)]
df_s['log_price'] = np.log1p(df_s['price_numeric'])

print(f"\n   Dataset: {len(df_s)} items")
print(f"   Price range: ${df_s['price_numeric'].min():.2f} - ${df_s['price_numeric'].max():.2f}")
print(f"   Median: ${df_s['price_numeric'].median():.2f}")

# 2. FEATURES
title_lower = df_s['title'].fillna('').str.lower()

known_brands = [
    'apple', 'samsung', 'sony', 'bose', 'garmin', 'dyson', 'lg', 'dell', 'hp',
    'lenovo', 'nike', 'adidas', 'microsoft', 'google', 'amazon', 'anker', 'jbl',
    'philips', 'panasonic', 'canon', 'nikon', 'logitech', 'corsair', 'razer',
    'beats', 'fitbit', 'gopro', 'intel', 'amd', 'nvidia', 'asus', 'acer',
    'roku', 'ring', 'nest', 'eufy', 'wyze', 'tp-link', 'netgear', 'linksys',
    'kitchenaid', 'ninja', 'instant pot', 'cuisinart', 'keurig', 'roomba',
    'dewalt', 'makita', 'bosch', 'milwaukee', 'craftsman', 'stanley',
    'lego', 'barbie', 'nerf', 'hasbro', 'mattel', 'fisher-price',
]

def extract_brand(title):
    t = title.lower()
    for brand in known_brands:
        if brand in t:
            return brand
    return 'unknown'

df_s['brand'] = title_lower.apply(extract_brand)

def extract_specs(title):
    t = title.lower()
    specs = {}
    gb = re.search(r'(\d+)\s*gb', t)
    specs['storage_gb'] = int(gb.group(1)) if gb else 0
    tb = re.search(r'(\d+)\s*tb', t)
    specs['storage_tb'] = int(tb.group(1)) if tb else 0
    inch = re.search(r'(\d+\.?\d*)\s*(?:inch|")', t)
    specs['screen_inch'] = float(inch.group(1)) if inch else 0
    mah = re.search(r'(\d+)\s*mah', t)
    specs['battery_mah'] = int(mah.group(1)) if mah else 0
    watt = re.search(r'(\d+)\s*(?:watt|w\b)', t)
    specs['wattage'] = int(watt.group(1)) if watt else 0
    pack = re.search(r'(?:pack of |(\d+)\s*-?\s*pack|set of (\d+)|(\d+)\s*count|(\d+)\s*pieces?)', t)
    if pack:
        nums = [g for g in pack.groups() if g]
        specs['pack_qty'] = int(nums[0]) if nums else 1
    else:
        specs['pack_qty'] = 1
    return specs

specs_data = title_lower.apply(extract_specs)
specs_df = pd.DataFrame(specs_data.tolist(), index=df_s.index)
df_s = pd.concat([df_s, specs_df], axis=1)

# Quality flags
df_s['is_premium'] = title_lower.str.contains(r'\b(premium|pro|professional|ultra|luxury|platinum|titanium|elite|advanced|deluxe)\b', regex=True).astype(int)
df_s['is_budget'] = title_lower.str.contains(r'\b(basic|mini|lite|cheap|budget|value|economy|starter|compact|portable)\b', regex=True).astype(int)
df_s['is_wireless'] = title_lower.str.contains(r'\b(wireless|bluetooth|wifi|wi-fi)\b', regex=True).astype(int)
df_s['is_waterproof'] = title_lower.str.contains(r'\b(waterproof|water.?resistant|ip\d+)\b', regex=True).astype(int)
df_s['is_organic'] = title_lower.str.contains(r'\b(organic|natural|eco|sustainable|biodegradable)\b', regex=True).astype(int)
df_s['has_led'] = title_lower.str.contains(r'\b(led|oled|amoled|lcd)\b', regex=True).astype(int)
df_s['is_smart'] = title_lower.str.contains(r'\b(smart|ai|alexa|google assistant|siri)\b', regex=True).astype(int)

# Accessory detection
ACC_PAT = r'\b(case|cover|protector|screen protector|cable|charger|cord|adapter|mount|holder|stand|strap|band|sleeve|pouch|skin|film|tempered glass|dock|cradle|clip|bracket|hub|dongle|converter|splitter|extender|replacement|refill|cartridge|filter|tip|nib|stylus|grip|pad|mat|insert|attachment|accessory|accessories)\b'
DEV_PAT = r'\b(phone|smartphone|laptop|notebook|computer|tablet|ipad|macbook|television|monitor|camera|drone|vacuum|washer|dryer|dishwasher|refrigerator|printer|scanner|projector|speaker|soundbar|headphones|earbuds|console|playstation|xbox|nintendo|guitar|piano|drum|treadmill|bicycle|scooter|robot|mixer|blender|oven|microwave|air purifier|humidifier|air conditioner)\b'
df_s['is_accessory'] = title_lower.str.contains(ACC_PAT, regex=True).astype(int)
df_s['is_main_device'] = title_lower.str.contains(DEV_PAT, regex=True).astype(int)

print(f"   Accessories: {df_s['is_accessory'].sum()}, Main devices: {df_s['is_main_device'].sum()}")

df_s['title_length'] = df_s['title'].str.len()
df_s['title_word_count'] = df_s['title'].str.split().str.len()

# 3. PRICE MEDIANS
cat_median = df_s.groupby('category')['log_price'].median().to_dict()
df_s['cat_median_log_price'] = df_s['category'].map(cat_median)
subcat_median = df_s.groupby('subcategory')['log_price'].median().to_dict()
df_s['subcat_median_log_price'] = df_s['subcategory'].map(subcat_median)
brand_median = df_s.groupby('brand')['log_price'].median().to_dict()
df_s['brand_median_log_price'] = df_s['brand'].map(brand_median)
cat_mean = df_s.groupby('category')['log_price'].mean().to_dict()
df_s['cat_mean_log_price'] = df_s['category'].map(cat_mean)
brand_mean = df_s.groupby('brand')['log_price'].mean().to_dict()
df_s['brand_mean_log_price'] = df_s['brand'].map(brand_mean)
global_median_log_price = float(df_s['log_price'].median())

# 4. BUILD FEATURES
print("   Building features...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_tfidf = tfidf.fit_transform(df_s['title'].fillna(''))

le_cat = LabelEncoder()
cat_encoded = le_cat.fit_transform(df_s['category'].fillna('unknown'))
le_subcat = LabelEncoder()
subcat_encoded = le_subcat.fit_transform(df_s['subcategory'].fillna('unknown'))
le_brand = LabelEncoder()
brand_encoded = le_brand.fit_transform(df_s['brand'])

numeric_features = df_s[[
    'storage_gb', 'storage_tb', 'screen_inch', 'battery_mah', 'wattage',
    'pack_qty', 'is_premium', 'is_budget', 'is_wireless', 'is_waterproof',
    'is_organic', 'has_led', 'is_smart',
    'is_accessory', 'is_main_device',
    'title_length', 'title_word_count',
    'cat_median_log_price', 'subcat_median_log_price', 'brand_median_log_price',
    'cat_mean_log_price', 'brand_mean_log_price',
]].values

X_cat = csr_matrix(np.column_stack([cat_encoded, subcat_encoded, brand_encoded]))
X_numeric = csr_matrix(numeric_features)
X = hstack([X_tfidf, X_cat, X_numeric])
y = df_s['log_price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Features: {X.shape[1]}")

# 5. TRAIN
print("\n   Training...")
model = GradientBoostingRegressor(
    n_estimators=800, max_depth=7, learning_rate=0.05,
    subsample=0.8, min_samples_split=10, min_samples_leaf=5,
    max_features=0.5, random_state=42, verbose=0
)
model.fit(X_train, y_train)

# 6. EVALUATE
y_pred_test = model.predict(X_test)
y_test_dollars = np.expm1(y_test)
y_pred_dollars = np.expm1(y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test_dollars, y_pred_dollars)
mape = np.median(np.abs((y_test_dollars - y_pred_dollars) / y_test_dollars)) * 100

print(f"\n   R2 (test):  {r2_test:.4f}")
print(f"   MAE:        ${mae_test:.2f}")
print(f"   MdAPE:      {mape:.1f}%")

# 7. SAVE
print("\n   Saving...")
joblib.dump(model, os.path.join(MODELS_DIR, 'model_price_predictor.pkl'))
joblib.dump(tfidf, os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
joblib.dump(le_cat, os.path.join(MODELS_DIR, 'le_category.pkl'))
joblib.dump(le_subcat, os.path.join(MODELS_DIR, 'le_subcategory.pkl'))
joblib.dump(le_brand, os.path.join(MODELS_DIR, 'le_brand.pkl'))

reference_prices = {
    'iphone 17 pro max': 1199, 'iphone 17 pro': 999, 'iphone 17': 799,
    'iphone 16 pro max': 1099, 'iphone 16 pro': 999, 'iphone 16': 799,
    'iphone 15 pro max': 999, 'iphone 15 pro': 899, 'iphone 15': 699,
    'iphone 14 pro max': 899, 'iphone 14 pro': 799, 'iphone 14': 599,
    'iphone 13 pro max': 749, 'iphone 13 pro': 649, 'iphone 13': 499,
    'iphone 12 pro max': 599, 'iphone 12 pro': 499, 'iphone 12': 399,
    'iphone se': 429,
    'galaxy s25 ultra': 1299, 'galaxy s25': 799,
    'galaxy s24 ultra': 1199, 'galaxy s24': 799,
    'galaxy s23 ultra': 999, 'galaxy s23': 699, 'galaxy s22': 549,
    'galaxy z fold': 1799, 'galaxy z flip': 999,
    'galaxy a54': 349, 'galaxy a15': 159, 'galaxy a14': 149,
    'pixel 9 pro': 999, 'pixel 9': 699,
    'pixel 8 pro': 899, 'pixel 8': 599,
    'pixel 7 pro': 699, 'pixel 7': 499,
    'macbook pro 16': 2499, 'macbook pro 14': 1999,
    'macbook air m4': 1099, 'macbook air m3': 999, 'macbook air m2': 899,
    'thinkpad x1 carbon': 1399, 'thinkpad t14': 899,
    'xps 15': 1299, 'xps 13': 999,
    'surface pro': 999, 'surface laptop': 899,
    'ipad pro 13': 1299, 'ipad pro 11': 999, 'ipad pro': 999,
    'ipad air': 599, 'ipad mini': 499, 'ipad 10': 349,
    'galaxy tab s9': 799, 'galaxy tab s8': 599,
    'airpods pro': 249, 'airpods max': 549, 'airpods 4': 129,
    'bose quietcomfort ultra': 429, 'bose quietcomfort': 349, 'bose 700': 379,
    'sony wh-1000xm5': 349, 'sony wf-1000xm5': 299,
    'beats studio pro': 349, 'beats solo 4': 199,
    'apple watch ultra': 799, 'apple watch series 10': 399,
    'apple watch series 9': 399, 'apple watch se': 249,
    'galaxy watch 7': 299, 'galaxy watch 6': 249,
    'fitbit sense 2': 249, 'fitbit versa 4': 199,
    'playstation 5': 499, 'ps5': 499,
    'xbox series x': 499, 'xbox series s': 299,
    'nintendo switch oled': 349, 'nintendo switch': 299, 'steam deck': 399,
    'dyson v15': 649, 'dyson v12': 449, 'dyson v8': 349,
    'dyson airwrap': 599, 'dyson supersonic': 429,
    'roomba j7': 599, 'roomba i7': 399, 'roomba j9': 799,
    'kitchenaid artisan': 379, 'kitchenaid mixer': 349,
    'instant pot duo': 89,
    'ring doorbell': 99, 'nest doorbell': 129, 'nest thermostat': 129,
    'gopro hero 12': 349, 'gopro hero 11': 299,
    'canon eos r6': 2499, 'sony a7 iv': 2499, 'nikon z6': 1799,
    'lg oled c3': 1299, 'lg oled c4': 1399, 'samsung neo qled': 1299,
    'sony bravia': 999,
}

metadata = {
    'known_brands': known_brands,
    'categories': le_cat.classes_.tolist(),
    'subcategories': le_subcat.classes_.tolist(),
    'brands': le_brand.classes_.tolist(),
    'r2_score': r2_test,
    'mae': mae_test,
    'median_pct_error': float(mape),
    'cat_median_log_price': cat_median,
    'subcat_median_log_price': subcat_median,
    'brand_median_log_price': brand_median,
    'cat_mean_log_price': cat_mean,
    'brand_mean_log_price': brand_mean,
    'global_median_log_price': global_median_log_price,
    'reference_prices': reference_prices,
}

with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("   Done!")
print("=" * 70)

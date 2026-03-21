import pandas as pd
import re
from collections import Counter

print("🔄 Chargement du dataset...")
df = pd.read_csv('amazon_data_clean.csv')
print(f"✅ {len(df):,} produits chargés\n")

# Définition des catégories et mots-clés
CATEGORIES = {
    'Electronics - TV & Video': [
        'tv', 'television', 'roku', 'fire stick', 'streaming stick', 'chromecast', 
        'apple tv', 'hdmi', 'monitor', 'display', 'projector', 'screen'
    ],
    'Electronics - Audio': [
        'headphone', 'earbuds', 'speaker', 'soundbar', 'audio', 'bluetooth speaker',
        'airpods', 'wireless earbuds', 'microphone', 'sound', 'beats', 'bose', 'sony headphones'
    ],
    'Electronics - Computers': [
        'laptop', 'computer', 'pc', 'desktop', 'tablet', 'ipad', 'macbook', 
        'chromebook', 'keyboard', 'mouse', 'webcam', 'monitor', 'ssd', 'hard drive',
        'ram', 'processor', 'graphics card'
    ],
    'Electronics - Smart Home': [
        'alexa', 'echo', 'smart home', 'smart plug', 'smart light', 'ring', 
        'nest', 'thermostat', 'security camera', 'doorbell', 'smart switch',
        'home automation', 'iot'
    ],
    'Electronics - Mobile & Accessories': [
        'phone', 'iphone', 'samsung galaxy', 'smartphone', 'phone case', 
        'screen protector', 'charger', 'charging cable', 'power bank', 
        'wireless charger', 'phone holder', 'mobile'
    ],
    'Electronics - Gaming': [
        'playstation', 'xbox', 'nintendo', 'gaming', 'controller', 'console',
        'ps5', 'ps4', 'switch', 'game', 'vr headset', 'gaming chair', 'gaming mouse'
    ],
    'Electronics - Cameras': [
        'camera', 'gopro', 'dslr', 'lens', 'tripod', 'photography', 
        'camcorder', 'action camera', 'webcam', 'canon', 'nikon', 'sony camera'
    ],
    'Electronics - Wearables': [
        'smartwatch', 'fitness tracker', 'apple watch', 'fitbit', 'garmin',
        'smart watch', 'activity tracker', 'heart rate monitor'
    ],
    'Home & Kitchen - Appliances': [
        'coffee maker', 'blender', 'toaster', 'microwave', 'air fryer', 
        'vacuum', 'dishwasher', 'refrigerator', 'oven', 'mixer', 'food processor',
        'instant pot', 'slow cooker', 'rice cooker'
    ],
    'Home & Kitchen - Furniture': [
        'chair', 'desk', 'table', 'sofa', 'bed', 'mattress', 'shelf', 
        'cabinet', 'dresser', 'bookshelf', 'furniture', 'couch'
    ],
    'Home & Kitchen - Decor': [
        'lamp', 'curtain', 'rug', 'pillow', 'blanket', 'wall art', 
        'picture frame', 'candle', 'vase', 'decoration', 'mirror'
    ],
    'Tools & Home Improvement': [
        'drill', 'screwdriver', 'hammer', 'saw', 'wrench', 'tool set',
        'power tool', 'ladder', 'paint', 'hardware', 'toolbox'
    ],
    'Sports & Outdoors': [
        'bike', 'bicycle', 'treadmill', 'yoga mat', 'dumbbell', 'weights',
        'camping', 'tent', 'backpack', 'hiking', 'fishing', 'sports equipment',
        'exercise', 'fitness equipment'
    ],
    'Toys & Games': [
        'toy', 'lego', 'puzzle', 'board game', 'doll', 'action figure',
        'kids toy', 'children toy', 'playset', 'stuffed animal'
    ],
    'Books & Media': [
        'book', 'kindle', 'e-reader', 'audiobook', 'magazine', 'comic',
        'novel', 'textbook', 'ebook reader'
    ],
    'Clothing & Accessories': [
        'shirt', 't-shirt', 'pants', 'jeans', 'dress', 'jacket', 'shoes',
        'sneakers', 'boots', 'socks', 'hat', 'gloves', 'scarf', 'watch',
        'sunglasses', 'belt', 'bag', 'backpack', 'wallet'
    ],
    'Beauty & Personal Care': [
        'shampoo', 'conditioner', 'soap', 'lotion', 'perfume', 'makeup',
        'skincare', 'hair dryer', 'electric shaver', 'toothbrush', 'cosmetics'
    ],
    'Health & Household': [
        'vitamins', 'supplements', 'first aid', 'thermometer', 'blood pressure',
        'scale', 'humidifier', 'air purifier', 'medical', 'health monitor'
    ],
    'Baby Products': [
        'baby', 'diaper', 'stroller', 'car seat', 'baby monitor', 'pacifier',
        'baby bottle', 'crib', 'baby carrier', 'infant'
    ],
    'Pet Supplies': [
        'dog', 'cat', 'pet', 'pet food', 'pet toy', 'leash', 'collar',
        'pet bed', 'aquarium', 'bird cage', 'pet carrier'
    ],
    'Office Products': [
        'printer', 'paper', 'pen', 'notebook', 'stapler', 'desk organizer',
        'office chair', 'filing cabinet', 'calculator', 'label maker', 'shredder'
    ],
    'Automotive': [
        'car', 'auto', 'vehicle', 'tire', 'car charger', 'dash cam',
        'car seat cover', 'car vacuum', 'car accessories', 'motor oil'
    ],
    'Garden & Outdoor': [
        'garden', 'plant', 'lawn mower', 'hose', 'grill', 'patio',
        'outdoor furniture', 'gardening tools', 'seeds', 'fertilizer'
    ],
    'Musical Instruments': [
        'guitar', 'piano', 'keyboard', 'drum', 'violin', 'ukulele',
        'music', 'instrument', 'amplifier', 'music stand'
    ]
}

def categorize_product(title):
    """
    Analyse le titre du produit et retourne la catégorie la plus appropriée
    """
    if pd.isna(title):
        return 'Uncategorized'
    
    title_lower = title.lower()
    
    # Compter les correspondances pour chaque catégorie
    category_scores = {}
    
    for category, keywords in CATEGORIES.items():
        score = 0
        for keyword in keywords:
            # Recherche de mots entiers pour éviter les faux positifs
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = len(re.findall(pattern, title_lower))
            score += matches
        
        if score > 0:
            category_scores[category] = score
    
    # Retourner la catégorie avec le score le plus élevé
    if category_scores:
        return max(category_scores, key=category_scores.get)
    else:
        return 'Other'

print("🔍 Analyse et catégorisation des produits...")
print("   (Cela peut prendre quelques minutes...)\n")

# Appliquer la catégorisation
df['category'] = df['title'].apply(categorize_product)

# Statistiques
print("=" * 80)
print("📊 RÉSULTATS DE LA CATÉGORISATION")
print("=" * 80)

category_counts = df['category'].value_counts()
print(f"\n📈 Distribution des catégories ({len(category_counts)} catégories trouvées):\n")

for category, count in category_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{category:40} : {count:6,} produits ({percentage:5.1f}%)")

print(f"\n{'Total':40} : {len(df):6,} produits (100.0%)")

# Produits non catégorisés
uncategorized = (df['category'] == 'Other').sum() + (df['category'] == 'Uncategorized').sum()
if uncategorized > 0:
    print(f"\n⚠️  Produits non catégorisés: {uncategorized:,} ({uncategorized/len(df)*100:.1f}%)")

# Exemples par catégorie
print("\n" + "=" * 80)
print("📋 EXEMPLES DE PRODUITS PAR CATÉGORIE (5 premiers)")
print("=" * 80)

for category in category_counts.head(10).index:
    print(f"\n🏷️  {category}:")
    examples = df[df['category'] == category]['title'].head(5)
    for i, title in enumerate(examples, 1):
        print(f"   {i}. {title[:70]}...")

# Sauvegarder le dataset avec les catégories
output_file = 'amazon_data_categorized.csv'
print(f"\n💾 Sauvegarde du dataset catégorisé dans '{output_file}'...")
df.to_csv(output_file, index=False)

print(f"\n✅ Catégorisation terminée avec succès!")
print(f"   - Fichier sauvegardé: {output_file}")
print(f"   - Nombre de catégories: {len(category_counts)}")
print(f"   - Taux de catégorisation: {((len(df) - uncategorized) / len(df) * 100):.1f}%")

import pandas as pd

print("🔄 Chargement du dataset catégorisé...")
df = pd.read_csv('amazon_data_categorized.csv')

print(f"✅ Dataset chargé: {len(df):,} lignes, {len(df.columns)} colonnes\n")

# Afficher les colonnes actuelles
print("📋 Colonnes actuelles:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print("\n" + "="*80)
print("🔧 NETTOYAGE DU DATASET")
print("="*80)

# 1. Supprimer la colonne 'price' originale
if 'price' in df.columns:
    df = df.drop('price', axis=1)
    print("✅ Colonne 'price' supprimée")
else:
    print("⚠️  Colonne 'price' non trouvée")

# 2. Renommer 'price_clean' en 'price'
if 'price_clean' in df.columns:
    df = df.rename(columns={'price_clean': 'price'})
    print("✅ Colonne 'price_clean' renommée en 'price'")
else:
    print("⚠️  Colonne 'price_clean' non trouvée")

# 3. Supprimer 'price_numeric'
if 'price_numeric' in df.columns:
    df = df.drop('price_numeric', axis=1)
    print("✅ Colonne 'price_numeric' supprimée")
else:
    print("⚠️  Colonne 'price_numeric' non trouvée")

print("\n" + "="*80)
print("📊 RÉSULTAT")
print("="*80)

print(f"\n📋 Nouvelles colonnes ({len(df.columns)} colonnes):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

# Afficher un aperçu
print(f"\n👀 Aperçu des données (5 premières lignes):\n")
print(df[['title', 'price', 'rating', 'category']].head().to_string())

# Sauvegarder le dataset final
output_file = 'final_dataset.csv'
print(f"\n💾 Sauvegarde du dataset final dans '{output_file}'...")
df.to_csv(output_file, index=False)

print(f"\n✅ Dataset final sauvegardé avec succès!")
print(f"   - Fichier: {output_file}")
print(f"   - Lignes: {len(df):,}")
print(f"   - Colonnes: {len(df.columns)}")
print(f"   - Taille: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

print("\n" + "="*80)
print("🎉 NETTOYAGE TERMINÉ!")
print("="*80)

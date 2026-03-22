import json
import os
def test_amazon_collection_quality():
    filename = '../amazon_crawlbase_collector/amazon_products_crawlbase.json'
    print(f"🔍 Début de l'audit QE pour : {filename}")
    print("-" * 50)

    # Vérification de l'existance du fichier
    if not os.path.exists(filename):
        print("❌ ÉCHEC : Le fichier n'existe pas. Le scraper n'a rien généré.")
        return
    else:
        print("✅ SUCCÈS : Fichier trouvé.")

    # Vérification si le fichier est un JSON valide ?
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("✅ SUCCÈS : Format JSON valide.")
    except Exception as e:
        print(f"❌ ÉCHEC : Le fichier est corrompu ou mal formé. Erreur : {e}")
        return

    # Vérification de la présence des données essentielles
    products = data.get("products", [])
    total_count = data.get("total_count", 0)

    if not products:
        print("❌ ÉCHEC : La liste des produits est vide.")
    else:
        print(f"✅ SUCCÈS : {len(products)} produits détectés (Annoncé : {total_count}).")

    # Audit de santé sur un échantillon (les 10 premiers)
    print("\n🔬 Audit profond sur un échantillon (Top 10) :")
    missing_info = 0
    for i, p in enumerate(products[:10]):
        name = p.get('name', 'NOM MANQUANT')
        asin = p.get('asin')
        price = p.get('price')

        if not asin or not price:
            print(f"  ⚠️ Produit {i + 1} incomplet : {name[:30]}... (ASIN: {asin}, Prix: {price})")
            missing_info += 1
        else:
            print(f"  OK - Produit {i + 1} : {asin} | {price}")

    if missing_info == 0:
        print("\n🏆 BILAN : Qualité de collecte parfaite sur l'échantillon !")
    else:
        print(f"\n⚠️ BILAN : {missing_info}/10 produits de l'échantillon ont des données manquantes.")


# Lancement du test
if __name__ == "__main__":
    test_amazon_collection_quality()
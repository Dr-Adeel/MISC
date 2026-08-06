import pandas as pd

def test_final_report_logic():
    filename = 'final_dataset.csv'
    print(f"📊 Audit de Cohérence du Rapport Final : {filename}")
    print("-" * 50)

    df = pd.read_csv(filename)
    # On nettoie les prix comme dans le test précédent pour avoir des vrais chiffres
    df['price_numeric'] = pd.to_numeric(df['price'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')

    # 1. Vérification du Prix Moyen (Indicateur financier n°1)
    avg_price = df['price_numeric'].mean()
    print(f"💰 CALCUL QE - Prix Moyen Global : ${avg_price:.2f}")
    print("   👉 Vérifie que le graphique 'Distribution des Prix' montre cette moyenne.")

    # 2. Vérification de la domination du Marché
    top_cat = df['category'].value_counts().idxmax()
    top_cat_count = df['category'].value_counts().max()
    print(f"🏆 CALCUL QE - Catégorie n°1 : {top_cat} ({top_cat_count} produits)")
    print("   👉 Vérifie que le graphique 'Top Catégories' affiche bien celle-ci en premier.")

    # 3. Vérification de la Satisfaction (Qualité)
    avg_rating = df['rating'].mean()
    print(f"⭐ CALCUL QE - Note Moyenne Globale : {avg_rating:.2f}/5")
    print("   👉 Vérifie que le rapport mentionne bien ce score de satisfaction.")

    # 4. Alerte sur les "Produits Fantômes" (Prix NaN)
    nan_prices = df['price_numeric'].isna().sum()
    if nan_prices > 0:
        print(f"⚠️ ALERTE QE : {nan_prices} produits n'ont pas de prix valide.")
        print(f"   L'analyse doit préciser qu'elle exclut ces {nan_prices} lignes pour être honnête.")

if __name__ == "__main__":
    test_final_report_logic()

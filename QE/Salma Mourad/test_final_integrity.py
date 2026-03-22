import pandas as pd
import os
def test_final_dataset_integrity():
    filename = 'final_dataset.csv'

    print(f"🛡️ Audit d'Intégrité Finale : {filename}")
    print("-" * 50)

    if not os.path.exists(filename):
        print(f"❌ ÉCHEC : Le fichier {filename} est introuvable.")
        return

    df = pd.read_csv(filename)

    # 1. Test des Valeurs Manquantes (Microbes)
    missing_total = df.isnull().sum().sum()
    if missing_total == 0:
        print("✅ SUCCÈS : Aucune donnée manquante (Dataset 100% complet).")
    else:
        print(f"⚠️ ATTENTION : {missing_total} cellules vides détectées.")

    # 2. Test de cohérence des Ratings (Notes)
    # On vérifie qu'aucune note n'est > 5 ou < 0
    invalid_ratings = df[(df['rating'] > 5) | (df['rating'] < 0)].shape[0]
    if invalid_ratings == 0:
        print("✅ SUCCÈS : Toutes les notes sont conformes (entre 0 et 5).")
    else:
        print(f"❌ ÉCHEC : {invalid_ratings} notes invalides détectées !")

    # 3. Test de cohérence des Prix
    # Un prix de 0.0 n'est pas forcément une erreur, mais c'est suspect
    zero_prices = df[df['price'] <= 0].shape[0]
    if zero_prices == 0:
        print("✅ SUCCÈS : Aucun prix nul ou négatif.")
    else:
        print(f"⚠️ INFO : {zero_prices} produits ont un prix de 0.0 (à vérifier).")

    # 4. Test des Doublons (ASIN uniques)
    duplicates = df.duplicated(subset=['asin']).sum()
    if duplicates == 0:
        print("✅ SUCCÈS : Aucun doublon détecté (Chaque ASIN est unique).")
    else:
        print(f"❌ ÉCHEC : {duplicates} doublons trouvés !")


# Lancement
if __name__ == "__main__":
    test_final_dataset_integrity()

import pandas as pd
import os
def test_categorization_logic():
    filename = 'final_dataset.csv'

    print(f"Audit QE de la Catégorisation : {filename}")
    print("-" * 50)

    if not os.path.exists(filename):
        print(f"Le fichier {filename} est introuvable.")
        return

    # Chargement des données
    df = pd.read_csv(filename)
    total_products = len(df)

    # Existance la colonne 'category' existe
    if 'category' not in df.columns:
        print("La colonne 'category' est manquante !")
        return
    else:
        print("Colonne 'category' détectée.")

    #Analyse de taux de catégorisation ("other" check)
    # On compte combien de produits sont dans 'other' ou vides
    other_count = df[df['category'].str.lower() == 'other'].shape[0]
    categorized_count = total_products - other_count
    rate = (categorized_count / total_products) * 100

    print(f"Statut du rangement :")
    print(f"   - Total produits : {total_products}")
    print(f"   - Produits bien classés : {categorized_count}")
    print(f"   - Produits en 'Other' (Non classés) : {other_count}")
    print(f"   - Taux de réussite : {rate:.2f}%")

    if rate > 50:
        print(f"Plus de la moitié des produits sont classés.")
    else:
        print(f"Le taux de classement est faible ({rate:.2f}%).")

    #Vérification de la diversité des boites
    unique_cats = df['category'].nunique()
    print(f"{unique_cats} catégories distinctes créées.")


# Lancement de test
if __name__ == "__main__":
    test_categorization_logic()
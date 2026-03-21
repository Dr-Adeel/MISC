# 📊 Notebook d'Analyse des Catégories - Guide d'Utilisation

## ✅ Fichier Créé

**Nom du fichier** : `category_analysis.ipynb`

## 📋 Contenu du Notebook

Le notebook contient **10 sections complètes** pour analyser en profondeur la colonne "category" :

### 1. **Configuration et Chargement des Données** 📥
- Import des bibliothèques (pandas, matplotlib, seaborn)
- Configuration des graphiques
- Chargement de `amazon_data_categorized.csv`

### 2. **Vue d'Ensemble des Catégories** 📋
- Nombre total de catégories (25)
- Distribution des produits par catégorie
- Pourcentages détaillés

### 3. **Visualisations des Catégories** 📊
- **Graphique en barres** : Top 15 des catégories
- **Graphique circulaire** : Répartition des Top 10

### 4. **Analyse des Prix par Catégorie** 💰
- Statistiques complètes (moyenne, médiane, min, max, écart-type)
- Graphique des prix moyens
- Boxplot de distribution des prix
- Identification des catégories les plus chères

### 5. **Analyse des Notes par Catégorie** ⭐
- Notes moyennes par catégorie
- Catégories les mieux notées
- Visualisation des notes

### 6. **Analyse de la Popularité** 🔥
- Achats moyens et totaux par catégorie
- Catégories les plus vendues
- Graphiques de popularité

### 7. **Analyse Prime par Catégorie** 🚀
- Pourcentage de produits Prime
- Comparaison entre catégories
- Visualisation du taux Prime

### 8. **Top Produits par Catégorie** 🏆
- Top 3 produits les plus populaires
- Pour les 5 principales catégories
- Détails complets (prix, note, achats)

### 9. **Résumé et Insights** 💡
- Insights clés automatiques :
  - Catégorie la plus représentée
  - Catégorie la plus chère
  - Catégorie la mieux notée
  - Catégorie la plus achetée
  - Catégorie avec le plus de Prime

### 10. **Export des Résultats** 💾
- Création d'un fichier CSV de résumé
- `category_analysis_summary.csv`
- Toutes les statistiques par catégorie

## 🚀 Comment Utiliser le Notebook

### Méthode 1 : Jupyter Notebook
```bash
# Ouvrir le notebook
jupyter notebook category_analysis.ipynb
```

### Méthode 2 : VS Code
1. Ouvrir VS Code
2. Installer l'extension "Jupyter"
3. Ouvrir `category_analysis.ipynb`
4. Cliquer sur "Run All" ou exécuter cellule par cellule

### Méthode 3 : JupyterLab
```bash
jupyter lab category_analysis.ipynb
```

## 📊 Visualisations Incluses

Le notebook génère automatiquement **8 graphiques** :

1. ✅ Barres horizontales - Top 15 catégories
2. ✅ Graphique circulaire - Top 10 catégories
3. ✅ Barres - Prix moyen par catégorie
4. ✅ Boxplot - Distribution des prix
5. ✅ Barres - Notes moyennes
6. ✅ Barres - Total des achats
7. ✅ Barres - Pourcentage Prime
8. ✅ Tableaux statistiques détaillés

## 📁 Fichiers Générés

Après exécution du notebook, vous aurez :

1. **category_analysis_summary.csv** - Résumé statistique
   - Nombre de produits par catégorie
   - Prix moyen
   - Note moyenne
   - Achats moyens
   - Pourcentage Prime

## 🎯 Insights Automatiques

Le notebook calcule automatiquement :

- 🏆 **Catégorie dominante** : Clothing & Accessories (4,105 produits)
- 💰 **Catégorie la plus chère** : Identifiée automatiquement
- ⭐ **Meilleure note** : Catégorie avec la note moyenne la plus élevée
- 🔥 **Plus populaire** : Basé sur le total des achats
- 🚀 **Taux Prime** : Catégorie avec le plus de produits Prime

## 💡 Exemples d'Utilisation

### Analyser une catégorie spécifique
```python
# Dans une nouvelle cellule
category_name = "Electronics - TV & Video"
cat_data = df[df['category'] == category_name]

print(f"Analyse de {category_name}:")
print(f"Nombre de produits: {len(cat_data):,}")
print(f"Prix moyen: ${cat_data['price_numeric'].mean():.2f}")
print(f"Note moyenne: {cat_data['rating'].mean():.2f}/5")
```

### Comparer deux catégories
```python
cat1 = df[df['category'] == "Electronics - Gaming"]
cat2 = df[df['category'] == "Toys & Games"]

print(f"Gaming - Prix moyen: ${cat1['price_numeric'].mean():.2f}")
print(f"Toys - Prix moyen: ${cat2['price_numeric'].mean():.2f}")
```

## 📈 Analyses Possibles

Avec ce notebook, vous pouvez :

- ✅ Identifier les catégories les plus rentables
- ✅ Comparer les performances entre catégories
- ✅ Trouver les opportunités de marché
- ✅ Analyser la stratégie Prime par catégorie
- ✅ Découvrir les tendances de prix
- ✅ Identifier les catégories à forte demande

## 🔄 Personnalisation

Vous pouvez facilement :

- Modifier les couleurs des graphiques
- Changer le nombre de catégories affichées (Top 10, Top 20, etc.)
- Ajouter de nouvelles métriques
- Créer des graphiques supplémentaires
- Filtrer par plage de prix

## ⚠️ Notes Importantes

- Le notebook utilise `amazon_data_categorized.csv`
- Assurez-vous que ce fichier est dans le même dossier
- Temps d'exécution : ~30 secondes pour toutes les cellules
- Les graphiques sont interactifs dans Jupyter

## 🎨 Personnalisation des Graphiques

Pour changer les couleurs :
```python
# Remplacer
colors = sns.color_palette("husl", n)
# Par
colors = sns.color_palette("Set2", n)  # ou "viridis", "plasma", etc.
```

---

**Bon travail avec votre analyse ! 📊🎉**

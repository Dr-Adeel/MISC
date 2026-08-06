# Système d'Analyse de Prix Amazon

Projet d'analyse de données et estimation de prix basé sur un dataset de 50,000+ produits Amazon.

## Structure du Projet

```
POO/
├── README.md                    # Documentation
├── final_dataset.csv            # Dataset principal (50,444 produits)
├── Projet_POO.ipynb            # Notebook principal avec architecture POO
├── global_analysis.ipynb        # Analyse statistique détaillée
├── generer_rapport_pdf.py      # Script de génération du rapport PDF
├── Rapport_Analysis.pdf         # Rapport d'analyse complet
├── rapport_images/              # Images pour le rapport
├── requirements.txt             # Dépendances Python
└── tests/                       # Tests unitaires
```

## Description du Dataset

Le fichier **final_dataset.csv** contient 50,444 produits Amazon avec 15 colonnes :

- **Informations produit** : `title`, `asin`, `url`, `category`
- **Prix** : `price`, `currency`
- **Évaluations** : `rating`, `reviews_count`
- **Services** : `isPrime`, `shippingMessage`, `sponsoredAd`
- **Données temporelles** : `collected_at`, `bought_info_last_month`
- **Métadonnées** : `image`, `source_domain`

**Qualité des données** : 99.9% de complétude, aucun doublon

## Installation

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Analyse POO (Notebook Principal)

Le notebook `Projet_POO.ipynb` contient le système d'estimation de prix basé sur une architecture orientée objet.

```bash
jupyter notebook Projet_POO.ipynb
```

**Fonctionnalités** :
- Estimation de prix intelligente
- Distinction automatique produits/accessoires
- Filtrage par catégorie
- Analyse statistique (médiane, IQR, outliers)
- Système de scoring de pertinence

**Exemples d'utilisation** :

```python
# Recherche de laptops
result = analyzer.get_price_estimate("Lenovo Laptop", product_type="main_product")

# Recherche d'accessoires
result = analyzer.get_price_estimate("iPhone Case", product_type="accessory")

# Recherche générale
result = analyzer.get_price_estimate("Sony Headphones")
```

### 2. Analyse Statistique Globale

Le notebook `global_analysis.ipynb` fournit une analyse statistique complète du dataset.

```bash
jupyter notebook global_analysis.ipynb
```

**Contenu** :
- Statistiques descriptives
- Distribution des prix
- Analyse par catégorie
- Satisfaction client
- Visualisations détaillées

### 3. Génération du Rapport PDF

```bash
python generer_rapport_pdf.py
```

Le script génère un rapport PDF professionnel avec :
- Statistiques complètes
- Graphiques et visualisations
- Analyses détaillées
- Architecture POO

## Architecture POO

### Classe PricePoint

Data Transfer Object qui encapsule les informations d'un produit :

```python
@dataclass
class PricePoint:
    name: str
    price: float
    url: str
    category: str
    rating: float
    reviews_count: int
    is_prime: bool
    product_type: str  # "main_product" ou "accessory"
```

**Fonctionnalités** :
- Détection automatique du type de produit
- Validation des données
- Représentation structurée

### Classe MarketAnalyzer

Moteur d'analyse intelligent pour l'estimation de prix :

```python
class MarketAnalyzer:
    def get_price_estimate(user_input, product_type, min_samples, iqr_factor)
    def display_results(result)
    def _match_product(product, criteria_list)
    def _calculate_relevance_score(product, criteria_list)
```

**Algorithmes** :
- Système de synonymes pour recherche flexible
- Filtrage IQR pour éliminer les outliers
- Scoring de pertinence basé sur la position des mots-clés
- Analyse par catégorie

## Statistiques du Dataset

| Métrique | Valeur |
|----------|--------|
| **Produits totaux** | 50,444 |
| **Catégories** | 25 |
| **Prix médian** | $24.99 |
| **Prix moyen** | $76.71 |
| **Note moyenne** | 4.48/5 |
| **Taux satisfaction (4+★)** | 95.8% |
| **Produits Prime** | 0.6% |
| **Complétude des données** | 99.9% |

### Top 5 Catégories

1. Other (34.3%)
2. Clothing & Accessories (8.1%)
3. Home & Kitchen - Furniture (7.6%)
4. Beauty & Personal Care (5.2%)
5. Sports & Outdoors (4.3%)

## Méthodologie

### Détection des Accessoires

Un produit est classé comme "accessoire" si son titre contient (dans les 5 premiers mots) :
- `case`, `cover`, `sleeve`, `bag`, `backpack`
- `charger`, `cable`, `adapter`, `cord`
- `screen protector`, `stand`, `holder`, `mount`
- `strap`, `band`, `skin`, `sticker`

### Filtrage des Prix (IQR)

```python
Q1 = percentile(25)
Q3 = percentile(75)
IQR = Q3 - Q1
Prix valides = [Q1 - 2*IQR, Q3 + 2*IQR]
```

Le facteur de 2.0 (au lieu de 1.5 classique) permet de conserver plus de données tout en éliminant les valeurs aberrantes extrêmes.

### Scoring de Pertinence

```python
score = Σ (100 / (position_mot_clé + 1))
```

Les produits avec mots-clés en début de titre sont mieux classés.

## Tests

Exécuter les tests unitaires :

```bash
python -m pytest tests/
```

Les tests couvrent :
- Chargement des données
- Détection des types de produits
- Estimation de prix
- Filtrage par catégorie
- Calculs statistiques

## Dépendances

Voir `requirements.txt` pour la liste complète. Principales dépendances :

- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- jupyter >= 1.0.0
- reportlab >= 3.6.0

## Résultats Clés

### Analyse de Prix

- **Zone de prix optimale** : $20-$50 (60% des produits)
- **Distribution** : Asymétrique avec queue longue vers les prix élevés
- **Outliers** : ~5% de produits hors de la fourchette IQR normale

### Satisfaction Client

- **Taux de satisfaction élevée (4+★)** : 95.8%
- **Note moyenne** : 4.48/5
- **Produits 5 étoiles** : 48.2%

### Catégories les Plus Rentables

1. Electronics - Wearables (médian: $71.78, note: 4.38)
2. Home & Kitchen - Furniture (médian: $63.99, note: 4.43)
3. Musical Instruments (médian: $61.47, note: 4.42)

## Limitations et Perspectives

### Limitations Actuelles

- Catégorie "Other" trop large (34.3%)
- Peu de produits Prime (0.6%)
- Pas d'analyse temporelle
- Données statiques (snapshot à un instant T)

### Perspectives Futures

- **Court terme**
  - Recherche fuzzy (tolérance aux fautes)
  - Filtrage par note minimale
  - Filtrage par fourchette de prix
  - Export des résultats

- **Moyen terme**
  - Interface web (Flask/Streamlit)
  - API REST
  - Analyse temporelle des tendances
  - Recommandations personnalisées

- **Long terme**
  - Machine Learning pour catégorisation
  - Prédiction de prix futurs
  - Détection d'anomalies
  - Intégration avec API Amazon réelle

## Auteur

Groupe Data Scientist

## Licence

Usage académique uniquement. Dataset Amazon utilisé à des fins éducatives.

---

Pour toute question, consulter les notebooks pour des exemples détaillés d'utilisation.

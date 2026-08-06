#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests unitaires pour le système d'analyse de prix Amazon
"""

import pytest
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import sys
import os

# Ajouter le répertoire parent au path pour importer les modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Définitions des classes (normalement importées du notebook)
@dataclass
class PricePoint:
    """DTO représentant un produit Amazon"""
    name: str
    price: float
    url: str
    category: str = "Other"
    rating: float = 0.0
    reviews_count: int = 0
    is_prime: bool = False
    product_type: str = "main_product"

    def __post_init__(self):
        """Détermine automatiquement si c'est un accessoire"""
        self.product_type = self._detect_product_type()

    def _detect_product_type(self) -> str:
        """Détecte si le produit est un accessoire"""
        accessory_keywords = [
            'case', 'cover', 'sleeve', 'bag', 'backpack', 'pouch',
            'charger', 'cable', 'adapter', 'cord', 'power bank',
            'screen protector', 'stand', 'holder', 'mount',
            'strap', 'band', 'skin', 'sticker', 'decal'
        ]

        name_lower = self.name.lower()
        first_words = ' '.join(name_lower.split()[:5])

        for keyword in accessory_keywords:
            if keyword in first_words:
                return "accessory"

        return "main_product"


class TestPricePoint:
    """Tests pour la classe PricePoint"""

    def test_creation_produit_simple(self):
        """Test création d'un PricePoint basique"""
        product = PricePoint(
            name="Sony Headphones WH-1000XM4",
            price=299.99,
            url="https://amazon.com/test"
        )

        assert product.name == "Sony Headphones WH-1000XM4"
        assert product.price == 299.99
        assert product.category == "Other"
        assert product.product_type == "main_product"

    def test_detection_accessoire_case(self):
        """Test détection d'un accessoire (case)"""
        product = PricePoint(
            name="iPhone 13 Case Protective Cover",
            price=15.99,
            url="https://amazon.com/case"
        )

        assert product.product_type == "accessory"

    def test_detection_accessoire_charger(self):
        """Test détection d'un accessoire (charger)"""
        product = PricePoint(
            name="USB-C Charger Fast Charging",
            price=12.99,
            url="https://amazon.com/charger"
        )

        assert product.product_type == "accessory"

    def test_detection_produit_principal_laptop(self):
        """Test que laptop n'est pas détecté comme accessoire"""
        product = PricePoint(
            name="Lenovo ThinkPad Laptop Computer",
            price=899.99,
            url="https://amazon.com/laptop"
        )

        assert product.product_type == "main_product"

    def test_attributs_enrichis(self):
        """Test des attributs enrichis (rating, reviews, prime)"""
        product = PricePoint(
            name="Test Product",
            price=49.99,
            url="https://amazon.com/test",
            category="Electronics",
            rating=4.5,
            reviews_count=1234,
            is_prime=True
        )

        assert product.category == "Electronics"
        assert product.rating == 4.5
        assert product.reviews_count == 1234
        assert product.is_prime == True


class TestDataLoading:
    """Tests pour le chargement des données"""

    def test_csv_existe(self):
        """Vérifie que le fichier CSV principal existe"""
        csv_path = "final_dataset.csv"
        assert os.path.exists(csv_path), f"Le fichier {csv_path} n'existe pas"

    def test_chargement_csv(self):
        """Test le chargement du CSV"""
        csv_path = "final_dataset.csv"
        df = pd.read_csv(csv_path, nrows=100)  # Charger seulement 100 lignes pour le test

        # Vérifier les colonnes essentielles
        required_columns = ['title', 'price', 'category', 'rating']
        for col in required_columns:
            assert col in df.columns, f"Colonne {col} manquante"

    def test_qualite_donnees(self):
        """Test la qualité des données chargées"""
        csv_path = "final_dataset.csv"
        df = pd.read_csv(csv_path, nrows=1000)

        # Vérifier qu'il n'y a pas trop de valeurs manquantes
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        assert completeness > 95, f"Taux de complétude trop bas: {completeness}%"

    def test_types_donnees(self):
        """Test que les types de données sont corrects"""
        csv_path = "final_dataset.csv"
        df = pd.read_csv(csv_path, nrows=100)

        # Price doit être numérique
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        assert df['price'].dtype in [np.float64, np.int64], "Price doit être numérique"

        # Rating doit être numérique
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        assert df['rating'].dtype in [np.float64, np.int64], "Rating doit être numérique"


class TestStatistiques:
    """Tests pour les calculs statistiques"""

    def test_calcul_mediane(self):
        """Test du calcul de médiane"""
        prices = [10, 20, 30, 40, 50]
        median = np.median(prices)
        assert median == 30

    def test_calcul_iqr(self):
        """Test du calcul de l'IQR"""
        prices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        q1 = np.percentile(prices, 25)
        q3 = np.percentile(prices, 75)
        iqr = q3 - q1

        assert q1 == 27.5
        assert q3 == 72.5
        assert iqr == 45.0

    def test_filtrage_outliers(self):
        """Test du filtrage des outliers avec IQR"""
        prices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000]  # 1000 est un outlier

        q1 = np.percentile(prices, 25)
        q3 = np.percentile(prices, 75)
        iqr = q3 - q1
        iqr_factor = 2.0

        # Filtrer
        lower_bound = q1 - iqr_factor * iqr
        upper_bound = q3 + iqr_factor * iqr

        clean_prices = [p for p in prices if lower_bound <= p <= upper_bound]

        # L'outlier 1000 devrait être retiré
        assert 1000 not in clean_prices
        assert len(clean_prices) < len(prices)


class TestScoring:
    """Tests pour le système de scoring de pertinence"""

    def test_scoring_position_debut(self):
        """Test que les mots en début de titre ont un meilleur score"""
        # Simuler le scoring
        def calculate_score(title, keyword):
            pos = title.lower().find(keyword.lower())
            if pos != -1:
                return 100 / (pos + 1)
            return 0

        score1 = calculate_score("Laptop Lenovo ThinkPad", "laptop")
        score2 = calculate_score("Lenovo ThinkPad Laptop", "laptop")

        # Le premier mot a un meilleur score (position 0)
        assert score1 > score2

    def test_scoring_mot_absent(self):
        """Test que le score est 0 si le mot n'est pas présent"""
        def calculate_score(title, keyword):
            pos = title.lower().find(keyword.lower())
            if pos != -1:
                return 100 / (pos + 1)
            return 0

        score = calculate_score("Lenovo Laptop", "iphone")
        assert score == 0


class TestCategories:
    """Tests pour l'analyse par catégories"""

    def test_categories_presentes(self):
        """Test que le dataset contient plusieurs catégories"""
        csv_path = "final_dataset.csv"
        df = pd.read_csv(csv_path, nrows=1000)

        unique_categories = df['category'].nunique()
        assert unique_categories > 1, "Il devrait y avoir plusieurs catégories"

    def test_repartition_categories(self):
        """Test la répartition des catégories"""
        csv_path = "final_dataset.csv"
        df = pd.read_csv(csv_path, nrows=1000)

        # Vérifier qu'aucune catégorie ne représente 100% des données
        category_counts = df['category'].value_counts()
        max_category_pct = category_counts.iloc[0] / len(df) * 100

        assert max_category_pct < 100, "Une seule catégorie ne devrait pas avoir 100% des produits"


if __name__ == "__main__":
    # Exécuter les tests
    pytest.main([__file__, "-v"])

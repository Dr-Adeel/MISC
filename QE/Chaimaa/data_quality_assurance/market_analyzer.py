import numpy as np
import unicodedata
from typing import List, Optional, Dict

class MarketAnalyzer:
    """
    Analyseur de marché Amazon intelligent.
    
    Permet de:
    - Estimer les prix de produits
    - Filtrer par type (produit principal vs accessoire)
    - Analyser par catégorie
    - Scorer la pertinence des résultats
    """
    
    def __init__(self, all_products):
        self.database = all_products
        
        # Dictionnaire de synonymes étendu
        self.category_synonyms = {
            # Informatique
            "laptop": ["computer", "notebook", "pc", "chromebook", "ultrabook", "macbook"],
            "pc": ["laptop", "computer", "desktop"],
            "ordinateur": ["laptop", "pc", "computer"],
            "screen": ["monitor", "display", "lcd"],
            
            # Mobile / Tablette
            "phone": ["smartphone", "cellphone", "mobile", "android", "iphone"],
            "smartphone": ["phone", "cellphone", "mobile"],
            "tablette": ["tablet", "ipad", "tab"],
            
            # Wearables
            "watch": ["smartwatch", "tracker", "gps", "wearable", "fitness"],
            "montre": ["watch", "smartwatch", "gps"],
            
            # Audio
            "headphones": ["earbuds", "earphones", "headset", "bluetooth", "airpods"],
            "ecouteur": ["earbuds", "headphones", "earphones"],
            
            # Accessoires
            "case": ["cover", "sleeve", "protection", "shell"],
            "sac": ["bag", "backpack", "case", "sleeve"],
            "chargeur": ["charger", "adapter", "cable", "cord", "power"],
            "souris": ["mouse"],
        }
    
    def _normalize(self, text: str) -> str:
        """Normalise le texte (enlève accents, met en minuscule)"""
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
        return text.lower().strip()
    
    def _expand_query(self, user_query: str) -> List[str]:
        """Transforme une requête utilisateur en critères de recherche"""
        words = self._normalize(user_query).split()
        expanded_criteria = []
        
        for word in words:
            if word in self.category_synonyms:
                options = [word] + self.category_synonyms[word]
                expanded_criteria.append("|".join(options))
            else:
                expanded_criteria.append(word)
        
        return expanded_criteria
    
    def _calculate_relevance_score(self, product, criteria_list: List[str]) -> float:
        """
        Calcule un score de pertinence pour un produit.
        Plus le match est au début du titre, plus le score est élevé.
        """
        name_norm = self._normalize(product.name)
        score = 0.0
        
        for criteria in criteria_list:
            if '|' in criteria:
                options = criteria.split('|')
                for opt in options:
                    pos = name_norm.find(opt)
                    if pos != -1:
                        # Score inversement proportionnel à la position
                        score += 100 / (pos + 1)
                        break
            else:
                pos = name_norm.find(criteria)
                if pos != -1:
                    score += 100 / (pos + 1)
        
        return score
    
    def _match_product(self, product, criteria_list: List[str]) -> bool:
        """Vérifie si un produit correspond aux critères de recherche"""
        name_norm = self._normalize(product.name)
        
        for criteria in criteria_list:
            if '|' in criteria:  # Concept (OR)
                options = criteria.split('|')
                if not any(opt in name_norm for opt in options):
                    return False
            else:  # Marque/modèle (AND)
                if criteria not in name_norm:
                    return False
        
        return True
    
    def get_price_estimate(
        self, 
        user_input: str, 
        product_type: Optional[str] = None,
        min_samples: int = 3,
        iqr_factor: float = 2.0
    ) -> Dict:
        """
        Estime le prix d'un produit basé sur la requête utilisateur.
        
        Args:
            user_input: Requête (ex: "Lenovo Laptop", "Chargeur iPhone")
            product_type: Filtrer par type:
                - None: tous les produits
                - "main_product": uniquement produits principaux
                - "accessory": uniquement accessoires
            min_samples: Nombre minimum de produits pour une estimation
            iqr_factor: Facteur pour le filtrage IQR (plus élevé = plus permissif)
        
        Returns:
            Dictionnaire avec les résultats d'analyse
        """
        # 1. Analyse de la requête
        criteria_list = self._expand_query(user_input)
        print(f"\n🧠 Requête: '{user_input}'")
        print(f"   Critères: {criteria_list}")
        if product_type:
            type_label = "🔧 Accessoires" if product_type == "accessory" else "📦 Produits principaux"
            print(f"   Filtre: {type_label}")
        
        # 2. Matching et scoring
        matches = []
        for product in self.database:
            # Filtrer par type si spécifié
            if product_type and product.product_type != product_type:
                continue
            
            # Vérifier le match
            if self._match_product(product, criteria_list):
                relevance_score = self._calculate_relevance_score(product, criteria_list)
                matches.append((product, relevance_score))
        
        # Trier par pertinence
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # 3. Vérification du nombre de résultats
        if len(matches) < min_samples:
            return {
                "success": False,
                "message": f"Pas assez de produits trouvés ({len(matches)}/{min_samples} requis).",
                "matches_count": len(matches)
            }
        
        # 4. Analyse statistique avec IQR
        prices = [p.price for p, _ in matches]
        q1 = np.percentile(prices, 25)
        q3 = np.percentile(prices, 75)
        iqr = q3 - q1
        
        # Filtrage des outliers (plus permissif avec iqr_factor=2.0)
        clean_matches = [
            (p, s) for p, s in matches 
            if (q1 - iqr_factor*iqr) <= p.price <= (q3 + iqr_factor*iqr)
        ]
        
        if not clean_matches:
            clean_matches = matches  # Fallback
        
        clean_prices = [p.price for p, _ in clean_matches]
        
        # 5. Analyse par catégorie
        category_breakdown = {}
        for product, score in clean_matches:
            cat = product.category
            if cat not in category_breakdown:
                category_breakdown[cat] = []
            category_breakdown[cat].append(product.price)
        
        category_stats = {
            cat: {
                "count": len(prices),
                "median": np.median(prices),
                "mean": np.mean(prices)
            }
            for cat, prices in category_breakdown.items()
        }
        
        # 6. Top résultats
        top_matches = clean_matches[:10]
        
        return {
            "success": True,
            "query": user_input,
            "product_type_filter": product_type,
            "total_matches": len(matches),
            "clean_matches": len(clean_matches),
            "fair_price": round(np.median(clean_prices), 2),
            "mean_price": round(np.mean(clean_prices), 2),
            "min_price": round(np.min(clean_prices), 2),
            "max_price": round(np.max(clean_prices), 2),
            "std_dev": round(np.std(clean_prices), 2),
            "categories": category_stats,
            "top_products": top_matches
        }
    
    def display_results(self, result: Dict):
        """Affiche joliment les résultats d'une estimation"""
        if not result["success"]:
            print(f"\n⚠️  {result['message']}")
            return
        
        print(f"\n" + "="*70)
        print(f"📊 RÉSULTATS POUR: {result['query']}")
        print("="*70)
        
        print(f"\n💰 ESTIMATION DE PRIX:")
        print(f"   Prix Recommandé (médian): ${result['fair_price']:.2f}")
        print(f"   Prix Moyen:               ${result['mean_price']:.2f}")
        print(f"   Fourchette:               ${result['min_price']:.2f} - ${result['max_price']:.2f}")
        print(f"   Écart-type:               ${result['std_dev']:.2f}")
        
        print(f"\n📦 ÉCHANTILLON:")
        print(f"   Produits trouvés:  {result['total_matches']}")
        print(f"   Produits analysés: {result['clean_matches']} (après filtrage outliers)")
        
        if result['categories']:
            print(f"\n🏷️  RÉPARTITION PAR CATÉGORIE:")
            for cat, stats in sorted(result['categories'].items(), 
                                    key=lambda x: x[1]['count'], 
                                    reverse=True)[:5]:
                print(f"   • {cat}")
                print(f"     {stats['count']} produits | Médian: ${stats['median']:.2f}")
        
        print(f"\n🏆 TOP 5 RÉSULTATS (par pertinence):")
        for i, (product, score) in enumerate(result['top_products'][:5], 1):
            print(f"   {i}. ${product.price:.2f} | {product.rating}⭐ ({product.reviews_count:,} avis)")
            print(f"      [{product.category}]")
            print(f"      {product.name[:80]}...")
            print()
        
        print("="*70)

print("✅ Classe MarketAnalyzer améliorée définie.")
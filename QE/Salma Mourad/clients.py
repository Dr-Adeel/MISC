import httpx
import urllib.parse
import logging
from typing import Dict, List
from interfaces import MarketProvider, MarketAnalyzer, PricingAgent
from schemas import PricePoint
from exceptions import MarketUnreachableError, InsufficientDataError, StatisticsError, AIAgentError

logger = logging.getLogger(__name__)


class AsyncScraperAPIProvider(MarketProvider):
    """
    Adaptateur asynchrone pour Scraper API.
    """
    BASE_URL = "http://api.scraperapi.com"

    def __init__(self, api_key: str, domain: str = "amazon.fr"):
        if not api_key:
            raise ValueError("Scraper API Key manquante dans le .env.")
        self.api_key = api_key
        self.domain = domain

    async def fetch_listings(self, product_name: str) -> List[PricePoint]:
        amazon_url = f"https://www.{self.domain}/s?k={urllib.parse.quote(product_name)}"
        params = {
            'api_key': self.api_key,
            'url': amazon_url,
            # 'render': 'true'  # Ajouter si Amazon bloque le rendu statique
        }

        logger.info(f"Recherche de '{product_name}' via Scraper API...")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()

                # FIX: Scraper API renvoie du HTML brut, pas du JSON.
                # La logique de parsing (BeautifulSoup ou l'API de parsing de Scraper API)
                # doit être implémentée ici avant de construire les PricePoint.
                # Exemple attendu une fois le parsing HTML implémenté :
                #
                #   soup = BeautifulSoup(response.text, "html.parser")
                #   raw_items = soup.select("div[data-component-type='s-search-result']")
                #   for item in raw_items:
                #       title = item.select_one("h2 span").text
                #       price_whole = item.select_one(".a-price-whole")
                #       ...
                #
                # Pour l'instant, on lève NotImplementedError pour échouer
                # explicitement plutôt que silencieusement.
                raise NotImplementedError(
                    "Le parsing HTML de la réponse Scraper API n'est pas encore implémenté. "
                    "Utiliser BeautifulSoup ou l'API de structured data de Scraper API."
                )

        except httpx.RequestError as e:
            raise MarketUnreachableError(source="Scraper API", reason=str(e))
        except httpx.HTTPStatusError as e:
            raise MarketUnreachableError(source="Scraper API", reason=f"Erreur HTTP {e.response.status_code}")

    async def ping(self) -> bool:
        """Sonde de santé pour /health"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    self.BASE_URL,
                    params={'api_key': self.api_key, 'url': 'http://httpbin.org/get'}
                )
                return resp.status_code == 200
        except Exception:
            # FIX: bare `except:` remplacé par `except Exception` — évite de capturer
            # BaseException (KeyboardInterrupt, SystemExit…).
            return False


class RealAnalyzerClient(MarketAnalyzer):
    """
    Adaptateur asynchrone pour l'équipe 2 (Data Science).
    Envoie les annonces validées à leur API pour récupérer les statistiques.
    """
    def __init__(self, api_url: str):
        self.api_url = api_url

    async def calculate_stats(self, listings: List[PricePoint]) -> Dict[str, float]:
        payload = [item.model_dump() for item in listings]

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(f"{self.api_url}/analyze", json=payload)
                response.raise_for_status()
                stats = response.json()

        # FIX: StatisticsError était levée à l'intérieur du try puis re-capturée
        # par `except Exception`, ce qui la double-encapsulait et perdait le message
        # original. On isole maintenant les erreurs réseau/HTTP du reste.
        except httpx.RequestError as e:
            raise StatisticsError(detail=f"Impossible de joindre l'API Data Science : {e}")
        except httpx.HTTPStatusError as e:
            raise StatisticsError(detail=f"Erreur HTTP de l'API Data Science : {e.response.status_code}")
        except Exception as e:
            raise StatisticsError(detail=f"Erreur de parsing de la réponse Data Science : {e}")

        # Vérification du contrat de données en dehors du try réseau
        required_keys = {"mean", "median", "min", "max", "count"}
        if not required_keys.issubset(stats.keys()):
            raise StatisticsError(
                detail=f"Données manquantes dans la réponse de l'équipe 2. Reçu : {list(stats.keys())}"
            )

        return stats


class RealAIAgentClient(PricingAgent):
    """
    Adaptateur pour l'équipe 3 (IA — Boukechouch / El Yousfi).
    Envoie les statistiques de marché et la stratégie au LLM.
    """
    def __init__(self, api_url: str):
        self.api_url = api_url

    async def get_price_advice(self, stats: Dict[str, float], strategy: str) -> Dict:
        payload = {
            "market_stats": stats,
            "strategy": strategy
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{self.api_url}/advise", json=payload)
                response.raise_for_status()
                ai_result = response.json()

        # FIX: même correction que RealAnalyzerClient — on sépare les erreurs réseau
        # de la validation du contrat pour éviter le double-encapsulage de AIAgentError.
        except httpx.RequestError as e:
            raise AIAgentError(detail=f"Impossible de joindre le serveur IA : {e}")
        except httpx.HTTPStatusError as e:
            raise AIAgentError(detail=f"Erreur HTTP du serveur IA : {e.response.status_code}")
        except Exception as e:
            raise AIAgentError(detail=f"Erreur de parsing de la réponse IA : {e}")

        # Vérification du contrat de données en dehors du try réseau
        required_keys = {"price", "justification", "confidence"}
        if not required_keys.issubset(ai_result.keys()):
            raise AIAgentError(
                detail=f"Format de réponse IA invalide. Clés reçues : {list(ai_result.keys())}"
            )

        return ai_result

    async def ping(self) -> bool:
        """Sonde de santé asynchrone pour l'Agent IA"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.api_url}/health")
                return resp.status_code == 200
        except Exception:
            # FIX: bare `except:` remplacé par `except Exception`
            return False

# ── Chargement de l'environnement en premier ────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── Imports standard ────────────────────────────────────────────────────────
import os
import time
import logging
from functools import lru_cache

# ── Imports tiers ────────────────────────────────────────────────────────────
from fastapi import FastAPI, Depends, HTTPException, Header, status

# ── Imports locaux ───────────────────────────────────────────────────────────
# FIX: tous les imports locaux regroupés ici, en haut du fichier.
# `from reporting import ...` était placé en bas du fichier (après les routes),
# ce qui viole PEP 8 et peut masquer des erreurs d'import au démarrage.
from controller import PriceController
from interfaces import MarketProvider, MarketAnalyzer, PricingAgent
from schemas import ProductRequest, PriceResponse, HealthResponse, ComponentStatus
from mocks import MockScraper, MockAnalyzer, MockAI
from clients import AsyncScraperAPIProvider, RealAnalyzerClient, RealAIAgentClient
from reporting import generate_system_report
from exceptions import (
    MarketUnreachableError,
    InsufficientDataError,
    DataContractError,
    StatisticsError,
    AIAgentError,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pricing API",
    description="Estimation de prix par analyse de marché + IA",
    version="1.1.0",
)


# ════════════════════════════════════════════════════════════════════════════
# INJECTION DE DÉPENDANCES — Remplacez les Mock* par les vraies classes ici.
# L'architecte (Oussama) n'aura qu'à modifier ces fonctions factory.
# ════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def get_scraper() -> MarketProvider:
    """Factory du scraper utilisant Scraper API."""
    api_key = os.getenv("SCRAPER_API_KEY")
    if not api_key:
        logger.warning("SCRAPER_API_KEY absente — basculement sur MockScraper.")
        return MockScraper()
    return AsyncScraperAPIProvider(api_key=api_key)


@lru_cache(maxsize=1)
def get_analyzer() -> MarketAnalyzer:
    """Factory de l'analyseur statistique de l'équipe 2."""
    analyzer_url = os.getenv("ANALYZER_API_URL")
    if not analyzer_url:
        # FIX: ValueError non gérée → 500 opaque pour le client.
        # On bascule sur le Mock avec un avertissement pour ne pas bloquer les autres équipes.
        # Remplacer par `raise` en production si le service réel est obligatoire.
        logger.warning("ANALYZER_API_URL absente — basculement sur MockAnalyzer.")
        return MockAnalyzer()
    return RealAnalyzerClient(api_url=analyzer_url)


@lru_cache(maxsize=1)
def get_ai_agent() -> PricingAgent:
    """Factory de l'agent IA (Équipe 3)."""
    ai_url = os.getenv("AI_AGENT_API_URL")
    if not ai_url:
        # FIX: même logique que get_analyzer — fallback gracieux en dev.
        logger.warning("AI_AGENT_API_URL absente — basculement sur MockAI.")
        return MockAI()
    return RealAIAgentClient(api_url=ai_url)


def get_controller(
    scraper: MarketProvider = Depends(get_scraper),
    analyzer: MarketAnalyzer = Depends(get_analyzer),
    ai_agent: PricingAgent = Depends(get_ai_agent),
) -> PriceController:
    """Assemble le contrôleur à partir des dépendances injectées."""
    return PriceController(scraper=scraper, analyzer=analyzer, ai_agent=ai_agent)


# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.post(
    "/estimate",
    response_model=PriceResponse,
    summary="Estimer le prix d'un produit",
    tags=["Pricing"],
)
async def estimate_price(
    request: ProductRequest,
    controller: PriceController = Depends(get_controller),
):
    """
    Pipeline complet : scraping → statistiques → conseil IA → réponse.
    Chaque erreur métier est convertie en réponse HTTP sémantique.
    """
    try:
        return await controller.process_request(request)

    except MarketUnreachableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except InsufficientDataError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except (DataContractError, StatisticsError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except AIAgentError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        )
    except Exception as exc:
        # Filet de sécurité : erreur inattendue → 500
        logger.exception("Erreur interne inattendue lors du traitement de '%s'", request.product_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne inattendue : {exc}",
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérifier l'état du système",
    tags=["Ops"],
)
async def health_check(
    scraper: MarketProvider = Depends(get_scraper),
    ai_agent: PricingAgent = Depends(get_ai_agent),
) -> HealthResponse:
    """
    Pinge chaque composant critique et retourne un rapport de santé détaillé.
    Utilisé par DevOps (Ayoub) pour les sondes Kubernetes / load-balancer.

    - 200 + status=healthy   → tout va bien
    - 200 + status=degraded  → au moins un composant répond mal
    """
    components: dict[str, ComponentStatus] = {}

    # ── Ping du scraper ──────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        scraper_ok = await scraper.ping()
        latency = round((time.monotonic() - t0) * 1000, 2)
        components["scraper"] = ComponentStatus(
            status="ok" if scraper_ok else "degraded",
            latency_ms=latency,
        )
    except Exception as exc:
        components["scraper"] = ComponentStatus(status="unreachable", detail=str(exc))

    # ── Ping de l'agent IA ───────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        ai_ok = await ai_agent.ping()
        latency = round((time.monotonic() - t0) * 1000, 2)
        components["ai_agent"] = ComponentStatus(
            status="ok" if ai_ok else "degraded",
            latency_ms=latency,
        )
    except Exception as exc:
        components["ai_agent"] = ComponentStatus(status="unreachable", detail=str(exc))

    # ── Statut global ────────────────────────────────────────────────────
    all_ok = all(c.status == "ok" for c in components.values())
    any_unreachable = any(c.status == "unreachable" for c in components.values())

    if all_ok:
        global_status = "healthy"
    elif any_unreachable:
        global_status = "unhealthy"
    else:
        global_status = "degraded"

    return HealthResponse(status=global_status, components=components)


# FIX: l'endpoint /admin/report est désormais protégé par un token d'API.
# Sans cette garde, n'importe quel appelant non authentifié peut déclencher
# une écriture sur le système de fichiers du conteneur.
# Le token est lu depuis la variable d'environnement ADMIN_TOKEN.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")


def verify_admin_token(x_admin_token: str = Header(..., alias="X-Admin-Token")) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="L'authentification admin n'est pas configurée (ADMIN_TOKEN manquant).",
        )
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token admin invalide.",
        )


@app.get("/admin/report", tags=["Admin"], dependencies=[Depends(verify_admin_token)])
async def trigger_report():
    """
    Génère un rapport de santé sur le disque du conteneur.
    Requiert le header `X-Admin-Token` avec la valeur de la variable ADMIN_TOKEN.
    """
    report_path = generate_system_report()
    return {"message": "Rapport généré avec succès.", "path": report_path}


# py -m uvicorn main:app --reload
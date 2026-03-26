"""
PricePulse — Pricing Agent
============================
The main orchestrator that combines:
  1.  ML prediction engine  (predict_price.py)
  2.  Gemini LLM reasoning  (llm_client.py)
  3.  Seller Persona logic   (personas.py)

Pipeline:
  product info → ML predict → build prompt → LLM refine → structured JSON
  (graceful fallback to ML-only when LLM is unavailable)
"""

import sys
import os

from config.personas import Persona, get_persona, list_personas, PERSONAS
from AI_engineer.prompts.pricing_prompts import build_pricing_prompt, build_fallback_response
from agents.llm_client import GeminiClient
from agents.semantic_filter import SemanticFilter

# ── Silent import of predict_price ──
os.environ['PRICEPULSE_SILENT'] = '1'
from predict_price import predict_price as ml_predict_price


class PricingAgent:
    """
    AI-powered pricing agent that blends ML predictions with
    LLM reasoning and seller persona strategies.

    Usage:
        agent = PricingAgent(persona="optimizer")
        result = agent.price("Apple MacBook Air M3", "Computers & Accessories")
    """

    def __init__(
        self,
        persona: str | Persona = "optimizer",
        llm_client: GeminiClient = None,
    ):
        """
        Args:
            persona:    Persona id string or Persona object.
            llm_client: Pre-configured GeminiClient (optional).
                        If None, will try to create one; if that fails → ML-only mode.
        """
        # Resolve persona
        if isinstance(persona, str):
            self.persona = get_persona(persona)
        else:
            self.persona = persona

        # Try to initialise LLM client
        self.llm: GeminiClient | None = llm_client
        self.llm_available = False

        if self.llm is None:
            try:
                self.llm = GeminiClient()
                self.llm_available = True
            except (ValueError, Exception):
                self.llm = None
                self.llm_available = False
        else:
            self.llm_available = True

        # Semantic filter (reuses same LLM client)
        self.semantic_filter: SemanticFilter | None = None
        if self.llm_available and self.llm is not None:
            self.semantic_filter = SemanticFilter(self.llm)

    # ── Core pricing method ────────────────────────────────────────

    def price(
        self,
        title: str,
        category: str,
        subcategory: str = None,
        use_llm: bool = True,
    ) -> dict:
        """
        Generate a complete pricing recommendation.

        Args:
            title:       Product title
            category:    Product category
            subcategory: Product subcategory (optional)
            use_llm:     Whether to use LLM enhancement (default True)

        Returns:
            Structured dict with pricing recommendation:
            {
                "suggested_price": float,
                "price_range": {"min": float, "max": float},
                "confidence": float,
                "strategy_name": str,
                "reasoning": str,
                "pricing_factors": [...],
                "seller_tips": [...],
                "persona": str,
                "ml_data": { ... },
                "product_validation": { ... },
                "_source": "llm" | "ml_fallback",
            }
        """
        # ── Step 1: Semantic Validation ────────────────────────────
        validation = self._validate_product(title, category, use_llm)

        # ── Step 2: ML prediction ──────────────────────────────────
        ml_result = ml_predict_price(title, category, subcategory)

        ml_price    = ml_result["predicted_price"]
        price_range = ml_result["price_range"]
        brand       = ml_result["brand_detected"]
        specs       = ml_result["specs"]
        flags       = ml_result["flags"]
        ref_source  = ml_result["reference_used"]
        ml_raw      = ml_result["ml_raw_price"]

        # Reference price (use predicted_price when ref was applied)
        ref_price = ml_price if ref_source else None

        # ── Step 3: Adjust for accessory detection ─────────────────
        # If semantic filter identified the product as an accessory,
        # reduce confidence and add a warning factor
        accessory_warning = False
        if validation.get("match_type") == "accessory":
            accessory_warning = True

        # ── Step 4: LLM Enhancement (if available) ─────────────────
        if use_llm and self.llm_available and self.llm is not None:
            try:
                recommendation = self._llm_price(
                    title, category, subcategory or category,
                    brand, ml_price, ref_price, ref_source,
                    price_range, specs, flags,
                )
            except Exception:
                recommendation = self._fallback_price(
                    ml_price, price_range, brand, category, ref_source
                )
        else:
            recommendation = self._fallback_price(
                ml_price, price_range, brand, category, ref_source
            )

        # ── Step 5: Apply accessory confidence penalty ─────────────
        if accessory_warning:
            recommendation["confidence"] = round(
                min(recommendation.get("confidence", 0.7), 0.4), 2
            )
            recommendation["pricing_factors"].append({
                "factor": "⚠️ Accessory Detected",
                "impact": "Low confidence",
                "explanation": (
                    f"This looks like an accessory, not a main product. "
                    f"Reason: {validation.get('reason', 'N/A')}"
                ),
            })

        # ── Step 6: Attach metadata ───────────────────────────────
        recommendation["persona"] = f"{self.persona.emoji} {self.persona.name}"
        recommendation["ml_data"] = {
            "ml_raw_price": ml_raw,
            "ml_adjusted_price": ml_price,
            "brand": brand,
            "category": category,
            "reference_used": ref_source,
            "specs": specs,
            "flags": flags,
        }
        recommendation["product_validation"] = validation

        return recommendation

    # ── Semantic product validation ────────────────────────────────

    def _validate_product(self, title: str, category: str, use_llm: bool) -> dict:
        """
        Validate that the product title is a genuine product for its category.
        Uses SemanticFilter (LLM) or keyword-based accessory detection (fallback).

        Returns:
            dict with is_match, confidence, match_type, reason
        """
        if use_llm and self.semantic_filter is not None:
            try:
                # Call LLM directly (bypass SemanticFilter's own keyword fallback
                # which compares category words vs title words — not useful here)
                from prompts.filter_prompts import build_filter_prompt
                system_prompt, user_prompt = build_filter_prompt(category, title)
                result = self.llm.generate_json(system_prompt, user_prompt)
                result = SemanticFilter._validate_single_result(result)
                result["is_match"] = (
                    result["confidence"] >= 0.5
                    and result["match_type"] in SemanticFilter.POSITIVE_MATCH_TYPES
                )
                return result
            except Exception:
                pass  # Fall through to keyword detection

        # ── Keyword-based product validation (no LLM) ──────────
        return self._keyword_product_validation(title)

    @staticmethod
    def _keyword_product_validation(title: str) -> dict:
        """
        Simple keyword-based validation to detect accessories vs real products.
        Works without LLM by checking for accessory-related keywords.
        """
        t_lower = title.lower()
        t_words = set(t_lower.split())

        # Accessory keywords
        accessory_kw = {
            "case", "cover", "protector", "screen", "cable", "charger",
            "adapter", "stand", "mount", "holder", "strap", "band",
            "sleeve", "pouch", "skin", "film", "tempered", "glass",
            "replacement", "refill", "cartridge", "stylus", "pen",
            "dock", "cradle", "clip", "hook", "bracket", "hinge",
        }

        found_accessory = accessory_kw & t_words

        if found_accessory:
            return {
                "is_match": False,
                "confidence": 0.75,
                "match_type": "accessory",
                "reason": f"Accessory keywords detected: {', '.join(found_accessory)}",
            }

        # If title is very short (< 3 words), flag as uncertain
        if len(t_words) < 3:
            return {
                "is_match": True,
                "confidence": 0.5,
                "match_type": "partial",
                "reason": "Very short title — limited validation possible",
            }

        # Default: assume it's a valid product
        return {
            "is_match": True,
            "confidence": 0.8,
            "match_type": "exact",
            "reason": "Product title looks valid (no accessory keywords detected)",
        }

    # ── LLM-based pricing ──────────────────────────────────────────

    def _llm_price(
        self,
        title, category, subcategory,
        brand, ml_price, ref_price, ref_source,
        price_range, specs, flags,
    ) -> dict:
        """Get LLM-enhanced pricing recommendation."""
        system_prompt, user_prompt = build_pricing_prompt(
            title=title,
            category=category,
            subcategory=subcategory,
            brand=brand,
            ml_price=ml_price,
            ref_price=ref_price,
            ref_source=ref_source,
            price_range=price_range,
            specs=specs,
            flags=flags,
            persona=self.persona,
        )

        result = self.llm.generate_json(system_prompt, user_prompt)

        # Validate & sanitize
        result = self._validate_response(result, ml_price, price_range)
        result["_source"] = "llm"
        return result

    # ── Fallback (ML-only) ─────────────────────────────────────────

    def _fallback_price(
        self, ml_price, price_range, brand, category, ref_source
    ) -> dict:
        """Build a structured recommendation without the LLM."""
        return build_fallback_response(
            ml_price=ml_price,
            price_range=price_range,
            persona=self.persona,
            brand=brand,
            category=category,
            ref_source=ref_source,
        )

    # ── Response validation ────────────────────────────────────────

    @staticmethod
    def _validate_response(
        result: dict,
        ml_price: float,
        price_range: tuple[float, float],
    ) -> dict:
        """Ensure LLM response has valid fields and sane values."""

        # Suggested price — must be a positive number
        try:
            sp = float(result.get("suggested_price", ml_price))
            if sp <= 0:
                sp = ml_price
        except (TypeError, ValueError):
            sp = ml_price

        # Sanity guard: LLM price shouldn't be wildly different from ML
        ratio = sp / ml_price if ml_price > 0 else 1
        if ratio < 0.2 or ratio > 5.0:
            sp = ml_price  # Revert to ML if LLM is hallucinating

        result["suggested_price"] = round(sp, 2)

        # Price range
        pr = result.get("price_range", {})
        try:
            pr_min = float(pr.get("min", price_range[0]))
            pr_max = float(pr.get("max", price_range[1]))
            if pr_min > pr_max:
                pr_min, pr_max = pr_max, pr_min
            result["price_range"] = {"min": round(pr_min, 2), "max": round(pr_max, 2)}
        except (TypeError, ValueError):
            result["price_range"] = {"min": price_range[0], "max": price_range[1]}

        # Confidence
        try:
            conf = float(result.get("confidence", 0.7))
            conf = max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            conf = 0.7
        result["confidence"] = round(conf, 2)

        # Strategy name
        result.setdefault("strategy_name", "AI Recommendation")

        # Reasoning
        result.setdefault("reasoning", "Price based on ML model with AI refinement.")

        # Pricing factors
        if not isinstance(result.get("pricing_factors"), list):
            result["pricing_factors"] = []

        # Seller tips
        if not isinstance(result.get("seller_tips"), list):
            result["seller_tips"] = ["Monitor competitors and adjust price regularly."]

        return result

    # ── Persona management ─────────────────────────────────────────

    def set_persona(self, persona_id: str):
        """Switch to a different persona."""
        self.persona = get_persona(persona_id)

    def get_status(self) -> dict:
        """Return current agent status for display."""
        return {
            "persona": f"{self.persona.emoji} {self.persona.name}",
            "persona_id": self.persona.id,
            "price_factor": self.persona.price_factor,
            "llm_available": self.llm_available,
            "semantic_filter": self.semantic_filter is not None,
            "llm_model": self.llm.model_name if self.llm else "N/A",
            "llm_stats": self.llm.get_usage_stats() if self.llm else None,
        }

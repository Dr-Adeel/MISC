"""
PricePulse — Semantic Filter Agent
====================================
Uses LLM to verify that product listing titles actually match
a user's search query. Detects accessories, wrong models, and
unrelated products.

Match types:
  - exact:     Direct match, same product
  - variant:   Same product line, different config (e.g., 128GB vs 256GB)
  - partial:   Related but not the same product
  - accessory: Case, charger, cable FOR the product (not the product itself)
  - unrelated: Completely different product
"""

import json
from typing import Optional

from prompts.filter_prompts import (
    build_filter_prompt,
    build_batch_filter_prompt,
    FILTER_JSON_SCHEMA,
)


class SemanticFilter:
    """
    Semantic product-match filter using the Gemini LLM.

    Determines whether a listing title genuinely matches a search query,
    filtering out accessories, wrong variants, and unrelated items.
    """

    # Minimum confidence to consider a match valid
    CONFIDENCE_THRESHOLD = 0.6

    # Match types that count as "genuine" product matches
    POSITIVE_MATCH_TYPES = {"exact", "variant"}

    def __init__(self, llm_client):
        """
        Args:
            llm_client: An initialized GeminiClient instance.
        """
        self.llm = llm_client

    # ── Single-item filter ─────────────────────────────────────────────

    def is_match(
        self,
        search_query: str,
        listing_title: str,
        threshold: float = None,
    ) -> dict:
        """
        Check if a single listing title matches the search query.

        Args:
            search_query:  What the user searched for (e.g., "iPhone 15 Pro")
            listing_title: The actual listing title to evaluate
            threshold:     Override confidence threshold (default 0.6)

        Returns:
            dict with keys:
              - is_match (bool)
              - confidence (float)
              - match_type (str): exact|variant|partial|accessory|unrelated
              - reason (str)
        """
        threshold = threshold or self.CONFIDENCE_THRESHOLD
        system_prompt, user_prompt = build_filter_prompt(
            search_query, listing_title
        )

        try:
            result = self.llm.generate_json(system_prompt, user_prompt)
            # Validate and normalise
            result = self._validate_single_result(result)
            # Apply threshold logic
            result["is_match"] = (
                result["confidence"] >= threshold
                and result["match_type"] in self.POSITIVE_MATCH_TYPES
            )
            return result

        except Exception as e:
            # Fallback: simple keyword overlap
            return self._keyword_fallback(search_query, listing_title, str(e))

    # ── Batch filter ───────────────────────────────────────────────────

    def filter_batch(
        self,
        search_query: str,
        listings: list[str],
        threshold: float = None,
    ) -> list[dict]:
        """
        Filter a batch of listings against a search query.

        Args:
            search_query: What the user searched for
            listings:     List of listing title strings
            threshold:    Override confidence threshold

        Returns:
            List of result dicts, one per listing (same structure as is_match)
        """
        if not listings:
            return []

        # For small batches (≤3), use single calls for better accuracy
        if len(listings) <= 3:
            return [
                self.is_match(search_query, title, threshold)
                for title in listings
            ]

        threshold = threshold or self.CONFIDENCE_THRESHOLD

        # Build batch prompt
        system_prompt, user_prompt = build_batch_filter_prompt(
            search_query, listings
        )

        try:
            raw = self.llm.generate_json(system_prompt, user_prompt)
            results_list = raw if isinstance(raw, list) else raw.get("results", [])

            results = []
            for i, item in enumerate(results_list):
                item = self._validate_single_result(item)
                item["is_match"] = (
                    item["confidence"] >= threshold
                    and item["match_type"] in self.POSITIVE_MATCH_TYPES
                )
                # Attach original title for reference
                if i < len(listings):
                    item["title"] = listings[i]
                results.append(item)

            return results

        except Exception as e:
            # Fall back to individual calls
            return [
                self.is_match(search_query, title, threshold)
                for title in listings
            ]

    # ── Convenience: get only matching titles ──────────────────────────

    def get_matching(
        self,
        search_query: str,
        listings: list[str],
        threshold: float = None,
    ) -> list[dict]:
        """Return only the listings that genuinely match."""
        all_results = self.filter_batch(search_query, listings, threshold)
        return [r for r in all_results if r.get("is_match")]

    # ── Internal helpers ───────────────────────────────────────────────

    @staticmethod
    def _validate_single_result(result: dict) -> dict:
        """Ensure result dict has all required fields with valid values."""
        valid_types = {"exact", "variant", "partial", "accessory", "unrelated"}

        return {
            "is_match": bool(result.get("is_match", False)),
            "confidence": float(result.get("confidence", 0.0)),
            "match_type": (
                result.get("match_type", "unrelated")
                if result.get("match_type") in valid_types
                else "unrelated"
            ),
            "reason": str(result.get("reason", "No reason provided")),
        }

    @staticmethod
    def _keyword_fallback(query: str, title: str, error_msg: str) -> dict:
        """
        Simple keyword-overlap fallback when the LLM is unavailable.
        """
        q_words = set(query.lower().split())
        t_words = set(title.lower().split())
        # Remove common stop words
        stop = {"the", "a", "an", "for", "with", "and", "or", "in", "of", "-", "&"}
        q_words -= stop
        t_words -= stop

        if not q_words:
            return {
                "is_match": False,
                "confidence": 0.0,
                "match_type": "unrelated",
                "reason": f"Keyword fallback (LLM error: {error_msg})",
            }

        overlap = q_words & t_words
        ratio = len(overlap) / len(q_words)

        # Accessory detection keywords
        accessory_kw = {
            "case", "cover", "protector", "screen", "cable", "charger",
            "adapter", "stand", "mount", "holder", "strap", "band",
            "sleeve", "pouch", "skin", "film", "tempered", "glass",
        }
        if accessory_kw & t_words:
            return {
                "is_match": False,
                "confidence": 0.7,
                "match_type": "accessory",
                "reason": f"Keyword fallback – accessory detected (LLM error: {error_msg})",
            }

        if ratio >= 0.8:
            match_type = "exact"
        elif ratio >= 0.5:
            match_type = "variant"
        elif ratio >= 0.3:
            match_type = "partial"
        else:
            match_type = "unrelated"

        return {
            "is_match": match_type in {"exact", "variant"},
            "confidence": round(ratio, 2),
            "match_type": match_type,
            "reason": f"Keyword fallback – {len(overlap)}/{len(q_words)} words match (LLM error: {error_msg})",
        }

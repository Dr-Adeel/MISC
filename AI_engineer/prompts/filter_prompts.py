"""
PricePulse — Semantic Filter Prompt Templates
===============================================
Prompts for determining if a product listing title
actually matches a search query (vs being an accessory,
unrelated product, or misleading listing).
"""


# ═══════════════════════════════════════════════════════════════
# JSON RESPONSE SCHEMA
# ═══════════════════════════════════════════════════════════════

FILTER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "is_match": {"type": "boolean", "description": "Whether the listing matches the search query"},
        "confidence": {"type": "number", "description": "Confidence level 0.0-1.0"},
        "match_type": {
            "type": "string",
            "enum": ["exact", "variant", "partial", "accessory", "unrelated"],
            "description": "Type of match: exact product, variant/color, partial match, accessory for it, or unrelated",
        },
        "reason": {"type": "string", "description": "Brief explanation of why it matches or not"},
    },
    "required": ["is_match", "confidence", "match_type", "reason"],
}


# ═══════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════

FILTER_SYSTEM_PROMPT = """You are a product matching expert for an e-commerce platform.

Your job is to determine whether a product listing title is a GENUINE match for a search query, 
or if it's something else (an accessory, a case, a cable, a different product entirely).

MATCH TYPES:
- "exact": The listing IS the searched product (correct brand, model, type)
- "variant": Same product but different color, size, storage, or minor variant
- "partial": Related product but not exactly what was searched (e.g., searched "iPhone 15" but listing is "iPhone 14")
- "accessory": An accessory FOR the searched product (case, cable, charger, screen protector)
- "unrelated": Completely different product

CRITICAL RULES:
1. "iPhone 15 case" is an ACCESSORY for "iPhone 15", NOT the phone itself
2. "USB cable for Samsung Galaxy" is an ACCESSORY, not the phone
3. Brand must match if specified in the search query
4. Model numbers must match if specified
5. A search for "laptop" should match actual laptops, not laptop bags or laptop stands

Respond with ONLY valid JSON. No markdown, no extra text."""


def build_filter_prompt(search_query: str, listing_title: str) -> tuple[str, str]:
    """
    Build the semantic filter prompt.

    Args:
        search_query: What the user is searching for
        listing_title: The product listing title to evaluate

    Returns:
        (system_prompt, user_prompt) tuple
    """
    user_prompt = f"""Determine if this listing matches the search query.

SEARCH QUERY: "{search_query}"
LISTING TITLE: "{listing_title}"

Is this listing the actual product being searched for?
Respond with ONLY valid JSON."""

    return FILTER_SYSTEM_PROMPT, user_prompt


def build_batch_filter_prompt(search_query: str, listings: list[str]) -> tuple[str, str]:
    """
    Build a batch semantic filter prompt for multiple listings at once.

    Args:
        search_query: What the user is searching for
        listings: List of product listing titles to evaluate

    Returns:
        (system_prompt, user_prompt) tuple
    """
    listings_text = "\n".join(
        f"  {i+1}. \"{title}\"" for i, title in enumerate(listings)
    )

    user_prompt = f"""Determine which of these listings match the search query.

SEARCH QUERY: "{search_query}"

LISTINGS:
{listings_text}

For EACH listing, determine if it's a genuine match.
Respond with ONLY a valid JSON array where each element has:
  {{"index": <number>, "is_match": <boolean>, "confidence": <0-1>, "match_type": "<type>", "reason": "<brief>"}}"""

    return FILTER_SYSTEM_PROMPT, user_prompt

"""
PricePulse — Pricing Prompts
============================
LLM prompt templates for price recommendations.
"""

# JSON schema for structured LLM responses
PRICING_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "suggested_price": {"type": "number"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 100},
        "reasoning": {"type": "string"},
        "pricing_factors": {"type": "array", "items": {"type": "string"}},
        "price_range": {
            "type": "object",
            "properties": {
                "low": {"type": "number"},
                "high": {"type": "number"}
            }
        },
        "seller_tips": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["suggested_price", "confidence", "reasoning", "pricing_factors"]
}

# Alias for backwards compatibility
PRICING_RESPONSE_SCHEMA = PRICING_JSON_SCHEMA


def _attr(obj, key, default=None):
    """Get attribute from dataclass or dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def build_pricing_prompt(title: str, category: str, subcategory: str = None,
                         brand: str = None, ml_price: float = 0,
                         ref_price: float = None, ref_source: str = None,
                         price_range: tuple = None, specs: dict = None,
                         flags: dict = None, persona=None) -> tuple[str, str]:
    """
    Build system and user prompts for pricing recommendation.
    
    Returns:
        (system_prompt, user_prompt)
    """
    if persona is None:
        persona = {"name": "Smart Optimizer", "description": "Balanced pricing", 
                   "price_factor": 1.0, "rules": [], "pricing_rules": []}
    
    p_name = _attr(persona, 'name', 'Smart Optimizer')
    p_desc = _attr(persona, 'description', 'Balanced pricing')
    p_factor = _attr(persona, 'price_factor', 1.0)
    p_rules = _attr(persona, 'pricing_rules', None) or _attr(persona, 'rules', [])
    
    system_prompt = f"""You are PricePulse, an expert Amazon pricing analyst. Your role is to recommend optimal prices for sellers.

PERSONA: {p_name}
STRATEGY: {p_desc}
PRICING APPROACH: {p_factor}x base price

RULES:
{chr(10).join(f"- {rule}" for rule in p_rules)}

You must respond with a JSON object containing:
- suggested_price: final price in USD (number)
- confidence: 0-100 confidence score (number)  
- reasoning: detailed explanation (string)
- pricing_factors: list of factors considered (array of strings)
- price_range: estimated price range (object with low/high)
- seller_tips: actionable advice for the seller (array of strings)

Be precise, data-driven, and aligned with the seller's strategy."""

    # Build specs info
    specs_info = ""
    if specs:
        spec_parts = []
        if specs.get('storage_gb'): spec_parts.append(f"{specs['storage_gb']}GB")
        if specs.get('storage_tb'): spec_parts.append(f"{specs['storage_tb']}TB")
        if specs.get('screen_inch'): spec_parts.append(f"{specs['screen_inch']}\"")
        if specs.get('battery_mah'): spec_parts.append(f"{specs['battery_mah']}mAh")
        if spec_parts:
            specs_info = f"\nSpecs: {', '.join(spec_parts)}"

    # Build flags info
    flags_info = ""
    if flags:
        flag_names = [k.replace('is_', '').replace('has_', '').upper() 
                      for k, v in flags.items() if v]
        if flag_names:
            flags_info = f"\nFeatures: {', '.join(flag_names)}"

    subcat_info = f"\nSubcategory: {subcategory}" if subcategory else ""
    brand_info = f"\nBrand: {brand}" if brand else ""
    ref_info = f"\nReference Price: ${ref_price:.2f} (source: {ref_source})" if ref_price else ""
    range_info = f"\nPrice Range: ${price_range[0]:.2f} - ${price_range[1]:.2f}" if price_range else ""
    
    user_prompt = f"""Analyze this product and recommend a price:

PRODUCT: {title}
CATEGORY: {category}{subcat_info}{brand_info}{specs_info}{flags_info}

ML MODEL PREDICTION: ${ml_price:.2f}{ref_info}{range_info}

Consider:
1. Brand value and recognition
2. Product tier (budget/mid/premium)
3. Category pricing norms
4. The ML prediction as a data-driven baseline
5. Your persona's pricing strategy

Respond with a JSON object only."""

    return system_prompt, user_prompt


def build_fallback_response(ml_price: float, price_range: tuple = None,
                           persona=None, brand: str = None,
                           category: str = None, ref_source: str = None) -> dict:
    """
    Build a fallback response when LLM is unavailable.
    Uses ML prediction adjusted by persona factor.
    """
    if persona is None:
        persona = {"name": "Smart Optimizer", "price_factor": 1.0, "rules": [], "id": "optimizer", "pricing_rules": []}
    
    price_factor = _attr(persona, 'price_factor', 1.0)
    p_name = _attr(persona, 'name', 'Optimizer')
    p_id = _attr(persona, 'id', 'optimizer')
    p_rules = _attr(persona, 'pricing_rules', None) or _attr(persona, 'rules', [])
    adjusted_price = ml_price * price_factor
    
    # Calculate price range if not provided
    if price_range is None:
        price_range = (ml_price * 0.85, ml_price * 1.15)
    
    strategy_names = {
        'maximize_sales': 'Competitive Undercut',
        'maximize_profit': 'Premium Positioning', 
        'optimizer': 'Revenue Optimization'
    }
    
    factors = [
        f"ML baseline: ${ml_price:.2f}",
        f"Persona adjustment: {price_factor}x",
        f"Strategy: {p_name}"
    ]
    if brand:
        factors.append(f"Brand: {brand}")
    if ref_source:
        factors.append(f"Reference: {ref_source}")
    
    return {
        "suggested_price": round(adjusted_price, 2),
        "confidence": 0.65,
        "reasoning": f"Based on ML prediction of ${ml_price:.2f}, adjusted by {price_factor-1:+.0%} for {p_name} strategy. ML model trained on 50,000+ Amazon products.",
        "pricing_factors": factors,
        "price_range": {
            "min": round(price_range[0], 2),
            "max": round(price_range[1], 2)
        },
        "seller_tips": list(p_rules)[:3],
        "strategy_name": strategy_names.get(p_id, 'Balanced'),
        "_source": "ml_fallback"
    }

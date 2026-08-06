"""
PricePulse — Agent System Tests
=================================
Tests the AI agent layer: Personas, Prompts, Pricing Agent, Semantic Filter.
Run:  python test_agent.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

# Suppress predict_price load banners
os.environ['PRICEPULSE_SILENT'] = '1'
from predict_price import predict_price


def separator(title):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


# ══════════════════════════════════════════════════════════════
# TEST 1: Persona System
# ══════════════════════════════════════════════════════════════
def test_personas():
    separator("TEST 1 — Persona System")
    from config.personas import get_persona, list_personas, PERSONAS

    # All 3 personas exist
    assert len(PERSONAS) == 3, f"Expected 3 personas, got {len(PERSONAS)}"
    print("   ✅ 3 personas defined")

    # Check each persona
    for pid in ["maximize_sales", "maximize_profit", "optimizer"]:
        p = get_persona(pid)
        assert p.id == pid
        assert p.name
        assert p.emoji
        assert p.goal
        assert p.system_prompt
        assert len(p.pricing_rules) >= 3
        assert 0.5 <= p.price_factor <= 2.0
        print(f"   ✅ {p.emoji} {p.name} — factor={p.price_factor}, rules={len(p.pricing_rules)}")

    # Price factors are correct
    sales = get_persona("maximize_sales")
    profit = get_persona("maximize_profit")
    optim = get_persona("optimizer")
    assert sales.price_factor < 1.0, "Sales should discount"
    assert profit.price_factor > 1.0, "Profit should markup"
    assert optim.price_factor == 1.0, "Optimizer should be neutral"
    print("   ✅ Price factor ordering: sales < optimizer < profit")

    # list_personas works
    summaries = list_personas()
    assert len(summaries) == 3
    print("   ✅ list_personas() returns 3 summaries")

    # Invalid persona raises
    try:
        get_persona("invalid_persona")
        assert False, "Should have raised KeyError"
    except KeyError:
        print("   ✅ Invalid persona raises KeyError")

    print("\n   🎉 Persona tests PASSED")


# ══════════════════════════════════════════════════════════════
# TEST 2: Prompt Templates
# ══════════════════════════════════════════════════════════════
def test_prompts():
    separator("TEST 2 — Prompt Templates")
    from config.personas import get_persona
    from prompts.pricing_prompts import (
        build_pricing_prompt,
        build_fallback_response,
        PRICING_JSON_SCHEMA,
    )
    from prompts.filter_prompts import (
        build_filter_prompt,
        build_batch_filter_prompt,
        FILTER_JSON_SCHEMA,
    )

    persona = get_persona("optimizer")

    # Pricing prompt
    system_p, user_p = build_pricing_prompt(
        title="Apple MacBook Air M3 256GB",
        category="Computers & Accessories",
        subcategory="Laptops",
        brand="Apple",
        ml_price=849.15,
        ref_price=899.0,
        ref_source="macbook air m3",
        price_range=(722.0, 976.0),
        specs={"storage_gb": 256, "storage_tb": 0, "screen_inch": 0, "battery_mah": 0, "wattage": 0, "pack_qty": 1},
        flags={"is_premium": 1, "is_budget": 0, "is_wireless": 0, "is_waterproof": 0, "is_organic": 0, "has_led": 0, "is_smart": 0},
        persona=persona,
    )
    assert "PricePulse" in system_p
    assert "MacBook" in user_p
    assert "$849.15" in user_p
    assert "JSON" in system_p
    print("   ✅ Pricing prompt builds correctly")
    print(f"      System prompt: {len(system_p)} chars")
    print(f"      User prompt:   {len(user_p)} chars")

    # Fallback response
    fallback = build_fallback_response(
        ml_price=849.15,
        price_range=(722.0, 976.0),
        persona=persona,
        brand="Apple",
        category="Computers & Accessories",
        ref_source="macbook air m3",
    )
    assert "suggested_price" in fallback
    assert "price_range" in fallback
    assert "confidence" in fallback
    assert "seller_tips" in fallback
    assert fallback["_source"] == "ml_fallback"
    print(f"   ✅ Fallback response: ${fallback['suggested_price']:.2f}")

    # Filter prompt (single)
    sys_f, usr_f = build_filter_prompt("iPhone 15 Pro", "Apple iPhone 15 Pro Max 256GB Blue")
    assert "iPhone 15 Pro" in usr_f
    assert "JSON" in sys_f
    print("   ✅ Filter prompt builds correctly")

    # Filter prompt (batch)
    sys_fb, usr_fb = build_batch_filter_prompt("iPhone 15 Pro", [
        "Apple iPhone 15 Pro 128GB",
        "iPhone 15 Pro Case Clear",
        "Samsung Galaxy S24 Ultra",
    ])
    assert "iPhone 15 Pro" in usr_fb
    assert "Case Clear" in usr_fb
    print("   ✅ Batch filter prompt builds correctly")

    # Schemas are valid
    assert "suggested_price" in str(PRICING_JSON_SCHEMA)
    assert "is_match" in str(FILTER_JSON_SCHEMA)
    print("   ✅ JSON schemas defined")

    print("\n   🎉 Prompt template tests PASSED")


# ══════════════════════════════════════════════════════════════
# TEST 3: Settings & Config
# ══════════════════════════════════════════════════════════════
def test_settings():
    separator("TEST 3 — Settings & Config")
    from config.settings import Settings

    assert Settings.GEMINI_MODEL, "Model name should be set"
    print(f"   ✅ Model:       {Settings.GEMINI_MODEL}")
    print(f"   ✅ Temperature: {Settings.LLM_TEMPERATURE}")
    print(f"   ✅ Max tokens:  {Settings.LLM_MAX_TOKENS}")
    print(f"   ✅ Retries:     {Settings.LLM_MAX_RETRIES}")

    configured = Settings.is_llm_configured()
    print(f"   ✅ LLM configured: {configured}")

    print("\n   🎉 Settings tests PASSED")


# ══════════════════════════════════════════════════════════════
# TEST 4: PricingAgent (ML Fallback Mode)
# ══════════════════════════════════════════════════════════════
def test_pricing_agent_fallback():
    separator("TEST 4 — PricingAgent (ML Fallback)")
    from agents.pricing_agent import PricingAgent

    test_products = [
        ("Apple MacBook Air M3 256GB", "Computers & Accessories", "maximize_sales"),
        ("Sony WH-1000XM5 Wireless Headphones", "Electronics", "maximize_profit"),
        ("Organic Green Tea 100 Bags", "Grocery & Gourmet Foods", "optimizer"),
    ]

    for title, category, persona_id in test_products:
        agent = PricingAgent(persona=persona_id)
        result = agent.price(title, category, use_llm=False)

        # Validate structure
        assert "suggested_price" in result, f"Missing suggested_price for {title}"
        assert "price_range" in result, f"Missing price_range for {title}"
        assert "confidence" in result, f"Missing confidence for {title}"
        assert "strategy_name" in result, f"Missing strategy_name for {title}"
        assert "reasoning" in result, f"Missing reasoning for {title}"
        assert "pricing_factors" in result, f"Missing pricing_factors for {title}"
        assert "seller_tips" in result, f"Missing seller_tips for {title}"
        assert "persona" in result, f"Missing persona for {title}"
        assert "ml_data" in result, f"Missing ml_data for {title}"
        assert "product_validation" in result, f"Missing product_validation for {title}"
        assert result["suggested_price"] > 0, f"Price should be positive for {title}"

        # Check validation structure
        val = result["product_validation"]
        assert "is_match" in val, f"Missing is_match in validation for {title}"
        assert "match_type" in val, f"Missing match_type in validation for {title}"
        assert "confidence" in val, f"Missing confidence in validation for {title}"

        from config.personas import get_persona
        p = get_persona(persona_id)
        v_icon = "✅" if val["is_match"] else "⚠️"
        print(f"   ✅ {p.emoji} {title[:35]:35s} → ${result['suggested_price']:.2f}  [{result['_source']}]  {v_icon} {val['match_type']}")

    print("\n   🎉 PricingAgent fallback tests PASSED")


# ══════════════════════════════════════════════════════════════
# TEST 5: Persona Price Factor Effects
# ══════════════════════════════════════════════════════════════
def test_persona_price_factors():
    separator("TEST 5 — Persona Price Factor Effects")
    from agents.pricing_agent import PricingAgent

    title = "Samsung Galaxy S24 Ultra 256GB"
    category = "Electronics"

    prices = {}
    for pid in ["maximize_sales", "maximize_profit", "optimizer"]:
        agent = PricingAgent(persona=pid)
        result = agent.price(title, category, use_llm=False)
        prices[pid] = result["suggested_price"]

    # Sales should be cheapest, Profit should be most expensive
    assert prices["maximize_sales"] < prices["optimizer"], \
        f"Sales (${prices['maximize_sales']}) should < Optimizer (${prices['optimizer']})"
    assert prices["maximize_profit"] > prices["optimizer"], \
        f"Profit (${prices['maximize_profit']}) should > Optimizer (${prices['optimizer']})"

    print(f"   🚀 Sales:     ${prices['maximize_sales']:.2f}")
    print(f"   🎯 Optimizer: ${prices['optimizer']:.2f}")
    print(f"   💎 Profit:    ${prices['maximize_profit']:.2f}")
    print(f"   📈 Spread:    ${prices['maximize_profit'] - prices['maximize_sales']:.2f}")

    print("\n   🎉 Price factor ordering CORRECT")


# ══════════════════════════════════════════════════════════════
# TEST 6: LLM Client (Initialization Only)
# ══════════════════════════════════════════════════════════════
def test_llm_client():
    separator("TEST 6 — LLM Client")
    from config.settings import Settings

    if not Settings.is_llm_configured():
        print("   ⏭️  Skipping — GEMINI_API_KEY not set")
        print("   ℹ️  Set it in .env to enable LLM tests")
        return

    from agents.llm_client import GeminiClient

    try:
        client = GeminiClient()
        print(f"   ✅ Client initialized: {client.model_name}")

        # Health check
        ok = client.health_check()
        if ok:
            print("   ✅ Health check passed — API is working")
        else:
            print("   ⚠️  Health check failed — API may have issues")

        stats = client.get_usage_stats()
        print(f"   ✅ Usage stats: {stats}")

    except Exception as e:
        print(f"   ⚠️  Client init failed: {e}")

    print("\n   🎉 LLM client test done")


# ══════════════════════════════════════════════════════════════
# TEST 7: Semantic Filter (keyword fallback)
# ══════════════════════════════════════════════════════════════
def test_semantic_filter_fallback():
    separator("TEST 7 — Semantic Filter (Keyword Fallback)")
    from agents.semantic_filter import SemanticFilter

    # Test the keyword fallback directly (no LLM needed)
    result = SemanticFilter._keyword_fallback(
        "iPhone 15 Pro", "Apple iPhone 15 Pro 256GB Black", "test"
    )
    assert result["is_match"] == True
    assert result["match_type"] in ("exact", "variant")
    print(f"   ✅ 'iPhone 15 Pro' → 'Apple iPhone 15 Pro 256GB': match={result['is_match']}, type={result['match_type']}")

    # Accessory detection
    result2 = SemanticFilter._keyword_fallback(
        "iPhone 15 Pro", "iPhone 15 Pro Case Clear Protective Cover", "test"
    )
    assert result2["match_type"] == "accessory"
    print(f"   ✅ 'iPhone 15 Pro' → 'iPhone 15 Pro Case Cover': match={result2['is_match']}, type={result2['match_type']}")

    # Unrelated product
    result3 = SemanticFilter._keyword_fallback(
        "iPhone 15 Pro", "Samsung Galaxy S24 Ultra 256GB", "test"
    )
    assert result3["is_match"] == False
    print(f"   ✅ 'iPhone 15 Pro' → 'Samsung Galaxy S24': match={result3['is_match']}, type={result3['match_type']}")

    print("\n   🎉 Semantic filter fallback tests PASSED")


# ══════════════════════════════════════════════════════════════
# TEST 8: Semantic Filter in Pricing Pipeline
# ══════════════════════════════════════════════════════════════
def test_semantic_filter_in_pipeline():
    separator("TEST 8 — Semantic Filter in Pricing Pipeline")
    from agents.pricing_agent import PricingAgent

    # Normal product → should validate fine
    agent = PricingAgent(persona="optimizer")
    result = agent.price("Sony WH-1000XM5 Wireless Headphones", "Electronics", use_llm=False)
    val = result["product_validation"]
    assert "match_type" in val
    print(f"   ✅ Normal product:    match_type={val['match_type']}, conf={val['confidence']:.0%}")

    # Accessory → should flag it and reduce confidence
    result2 = agent.price("iPhone 15 Pro Case Clear Cover Protector", "Electronics", use_llm=False)
    val2 = result2["product_validation"]
    assert val2["match_type"] == "accessory", f"Expected accessory, got {val2['match_type']}"
    assert result2["confidence"] <= 0.4, f"Accessory confidence should be ≤ 0.4, got {result2['confidence']}"
    # Check that an accessory warning factor was added
    factor_names = [f.get("factor", "") if isinstance(f, dict) else str(f) for f in result2.get("pricing_factors", [])]
    assert any("Accessory" in f for f in factor_names), "Missing accessory warning in pricing_factors"
    print(f"   ✅ Accessory product: match_type={val2['match_type']}, conf={result2['confidence']:.0%} (capped)")
    print(f"      → Accessory warning added to pricing_factors ✅")

    print("\n   🎉 Semantic filter pipeline integration PASSED")


# ══════════════════════════════════════════════════════════════
# TEST 9: Full Agent + LLM Integration (if API key available)
# ══════════════════════════════════════════════════════════════
def test_full_llm_integration():
    separator("TEST 9 — Full LLM Integration")
    from config.settings import Settings

    if not Settings.is_llm_configured():
        print("   ⏭️  Skipping — GEMINI_API_KEY not set")
        print("   ℹ️  Set it in .env to test full LLM integration")
        return

    from agents.pricing_agent import PricingAgent

    title = "Apple MacBook Air M3 256GB"
    category = "Computers & Accessories"

    for pid in ["maximize_sales", "maximize_profit", "optimizer"]:
        try:
            agent = PricingAgent(persona=pid)
            result = agent.price(title, category, use_llm=True)

            source = result.get("_source", "?")
            from config.personas import get_persona
            p = get_persona(pid)
            print(f"   ✅ {p.emoji} {p.name:20s} → ${result['suggested_price']:.2f}  [{source}]  conf={result['confidence']:.0%}")
            print(f"      Strategy: {result.get('strategy_name', 'N/A')}")
            print(f"      Reasoning: {result.get('reasoning', 'N/A')[:80]}...")
        except Exception as e:
            print(f"   ⚠️  {pid}: {e}")

    print("\n   🎉 Full LLM integration test done")


# ══════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "🧪 " * 20)
    print("  PRICEPULSE AGENT SYSTEM — TEST SUITE")
    print("🧪 " * 20)

    tests = [
        ("Persona System", test_personas),
        ("Prompt Templates", test_prompts),
        ("Settings & Config", test_settings),
        ("PricingAgent Fallback", test_pricing_agent_fallback),
        ("Persona Price Factors", test_persona_price_factors),
        ("LLM Client", test_llm_client),
        ("Semantic Filter Fallback", test_semantic_filter_fallback),
        ("Semantic Filter Pipeline", test_semantic_filter_in_pipeline),
        ("Full LLM Integration", test_full_llm_integration),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n   ❌ FAILED: {name}")
            print(f"      Error: {e}")
            failed += 1

    separator("TEST RESULTS")
    print(f"\n   ✅ Passed:  {passed}")
    print(f"   ❌ Failed:  {failed}")
    total = passed + failed
    print(f"   📊 Total:   {total}")
    print(f"\n{'═' * 60}\n")


if __name__ == "__main__":
    main()

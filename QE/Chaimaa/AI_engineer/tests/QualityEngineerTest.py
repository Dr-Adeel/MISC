import sys
import os
import types

# Fix import path (so "config", "agents", etc. work)
#this part took the longest to fix cs either this or install a bunch of stuff that will cause issues later
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock google.genai (to avoid Gemini dependency)
google_module = types.ModuleType("google")
genai_module = types.ModuleType("genai")
types_module = types.ModuleType("types")

genai_module.types = types_module
google_module.genai = genai_module

sys.modules["google"] = google_module
sys.modules["google.genai"] = genai_module
sys.modules["google.genai.types"] = types_module

# Mock ML prediction (to avoid missing model file)
def fake_predict_price(title, category, subcategory=None):
    return {
        "predicted_price": 100.0,
        "price_range": (80.0, 120.0),
        "brand_detected": "TestBrand",
        "specs": {},
        "flags": [],
        "reference_used": None,
        "ml_raw_price": 100.0,
    }

import sys
sys.modules["predict_price"] = type(sys)("predict_price")
sys.modules["predict_price"].predict_price = fake_predict_price

# NOW import agent
from agents.pricing_agent import PricingAgent



# NOW TESTS


def test_ai_output_structure():
    agent = PricingAgent()
    result = agent.price("iPhone 13", "Electronics", use_llm=False)

    assert "suggested_price" in result
    assert "price_range" in result
    assert "confidence" in result
    assert "reasoning" in result


def test_price_not_negative():
    agent = PricingAgent()
    result = agent.price("iPhone 13", "Electronics", use_llm=False)

    assert result["suggested_price"] > 0


def test_price_not_extreme():
    agent = PricingAgent()
    result = agent.price("iPhone 13", "Electronics", use_llm=False)

    ml_price = result["ml_data"]["ml_adjusted_price"]
    ai_price = result["suggested_price"]

    assert 0.2 * ml_price <= ai_price <= 5 * ml_price


def test_persona_changes_price():
    agent_fast = PricingAgent(persona="maximize_sales")
    agent_profit = PricingAgent(persona="maximize_profit")

    result_fast = agent_fast.price("iPhone 13", "Electronics", use_llm=False)
    result_profit = agent_profit.price("iPhone 13", "Electronics", use_llm=False)

    assert result_fast["suggested_price"] != result_profit["suggested_price"]


def test_fallback_mode():
    agent = PricingAgent()
    result = agent.price("iPhone 13", "Electronics", use_llm=False)

    assert result["_source"] == "ml_fallback"


def test_accessory_detection():
    agent = PricingAgent()
    result = agent.price("iPhone case", "Electronics", use_llm=False)

    assert result["product_validation"]["match_type"] == "accessory"

def test_invalid_persona():
    try:
        PricingAgent(persona="random")
        assert False  # should not reach here
    except KeyError:
        assert True

def test_empty_title():
    agent = PricingAgent()
    result = agent.price("", "Electronics", use_llm=False)

    assert result is not None

def test_gibberish_input():
    agent = PricingAgent()
    result = agent.price("asdfghjkl", "Electronics", use_llm=False)

    assert result["product_validation"]["confidence"] < 0.8

def test_very_long_title():
    agent = PricingAgent()
    long_title = "iphone " * 1000

    result = agent.price(long_title, "Electronics", use_llm=False)

    assert result["suggested_price"] > 0

def test_invalid_category():
    agent = PricingAgent()
    result = agent.price("iPhone 13", "INVALID_CATEGORY", use_llm=False)

    assert result is not None
def test_extreme_ml_values():
    agent = PricingAgent()

    # simulate extreme ML output
    def bad_predict(*args, **kwargs):
        return {
            "predicted_price": 1000000.0,
            "price_range": (1.0, 2000000.0),
            "brand_detected": "Test",
            "specs": {},
            "flags": [],
            "reference_used": None,
            "ml_raw_price": 1000000.0,
        }

    import sys
    sys.modules["predict_price"].predict_price = bad_predict

    result = agent.price("iPhone", "Electronics", use_llm=False)

    assert result["suggested_price"] < 2000000

def test_false_accessory_detection():
    agent = PricingAgent()

    result = agent.price("iPhone 13 with case included", "Electronics", use_llm=False)

    assert result["product_validation"]["match_type"] != "accessory"

def test_price_range_consistency():
    agent = PricingAgent()
    result = agent.price("iPhone 13", "Electronics", use_llm=False)

    pr = result["price_range"]

    assert pr["min"] <= result["suggested_price"] <= pr["max"]

def test_main_product_with_accessory_words():
    agent = PricingAgent()

    result = agent.price("Samsung Galaxy phone with charger", "Electronics", use_llm=False)

    assert result["product_validation"]["match_type"] != "accessory"

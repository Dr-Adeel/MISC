
import sys
import types
# Fake the google.genai module cs I'm too lazy to install it and it causes import errors and headaches
# This allows us to test the fallback mode without needing the actual LLM client or API keys, fake it till you make it babyyy  
google_module = types.ModuleType("google")
genai_module = types.ModuleType("genai")
types_module = types.ModuleType("types")

genai_module.types = types_module
google_module.genai = genai_module

sys.modules["google"] = google_module
sys.modules["google.genai"] = genai_module
sys.modules["google.genai.types"] = types_module

import pytest
from AI_engineer.agents.pricing_agent import PricingAgent
from config.settings import Settings 


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
    

    agent_fast = PricingAgent(persona="fast_sale")
    agent_profit = PricingAgent(persona="max_profit")

    result_fast = agent_fast.price("iPhone 13", "Electronics", use_llm=False)
    result_profit = agent_profit.price("iPhone 13", "Electronics", use_llm=False)

    assert result_fast["suggested_price"] != result_profit["suggested_price"]
def test_persona_changes_price():
    

    agent_fast = PricingAgent(persona="fast_sale")
    agent_profit = PricingAgent(persona="max_profit")

    result_fast = agent_fast.price("iPhone 13", "Electronics", use_llm=False)
    result_profit = agent_profit.price("iPhone 13", "Electronics", use_llm=False)

    assert result_fast["suggested_price"] != result_profit["suggested_price"]

def test_fallback_mode():
    agent = PricingAgent()

    result = agent.price("iPhone 13", "Electronics", use_llm=False)

    assert result["_source"] == "ml_fallback"
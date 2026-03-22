"""
PricePulse — Unified CLI Application
======================================
Main entry point combining:
  • ML Price Prediction
  • AI Pricing Agent (Gemini LLM + Personas)
  • Semantic Filtering

Usage:
  python app.py                    → Interactive menu
  python app.py --quick "iPhone 15 Pro" "Electronics"
  python app.py --persona optimizer "MacBook Air M3" "Computers & Accessories"
"""

import sys
import os
import json
import argparse

# ── Force UTF-8 output on Windows (emojis in terminal) ────────
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Suppress predict_price load banners ────────────────────────
os.environ['PRICEPULSE_SILENT'] = '1'
from predict_price import (
    predict_price as ml_predict,
    categories as CATEGORIES,
)

from config.personas import get_persona, list_personas, PERSONAS
from config.settings import Settings


# ═══════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════

BANNER = r"""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║    💲  P R I C E   P U L S E   v 2 . 0                   ║
║    ─────────────────────────────────                      ║
║    AI-Powered E-Commerce Pricing Engine                   ║
║                                                           ║
║    ML Prediction  ·  Gemini AI  ·  Seller Personas        ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""

SEPARATOR = "═" * 60


def print_banner():
    print(BANNER)


def print_section(title, emoji="📌"):
    print(f"\n{SEPARATOR}")
    print(f" {emoji}  {title}")
    print(SEPARATOR)


def print_categories():
    """Display available categories."""
    print_section("Available Categories", "📂")
    for i, cat in enumerate(CATEGORIES, 1):
        print(f"   {i:>2}. {cat}")
    print()


# ═══════════════════════════════════════════════════════════════
# ML-ONLY QUICK PREDICT
# ═══════════════════════════════════════════════════════════════

def quick_predict(title: str, category: str, subcategory: str = None):
    """Run ML prediction and display results (no LLM needed)."""
    result = ml_predict(title, category, subcategory)

    print_section("ML Price Prediction", "💰")
    print(f"\n   📦 Product:  {title[:80]}{'...' if len(title) > 80 else ''}")
    print(f"   🏷️  Brand:    {result['brand_detected']}")
    print(f"\n   ┌────────────────────────────────────────┐")
    print(f"   │  💵 Price:  ${result['predicted_price']:.2f}")
    print(f"   │  📊 Range:  ${result['price_range'][0]:.2f} — ${result['price_range'][1]:.2f}")
    print(f"   └────────────────────────────────────────┘")

    # Specs
    active_specs = {k: v for k, v in result['specs'].items()
                    if not k.startswith('_') and k != 'pack_qty'
                    and isinstance(v, (int, float)) and v > 0}
    if active_specs or result['specs']['pack_qty'] > 1:
        print(f"\n   🔧 Specs:")
        for k, v in active_specs.items():
            print(f"      • {k.replace('_', ' ').title()}: {v}")
        if result['specs']['pack_qty'] > 1:
            print(f"      • Pack: {result['specs']['pack_qty']}")

    # Flags
    active_flags = [k.replace('is_', '').replace('has_', '').replace('_', ' ').title()
                    for k, v in result['flags'].items() if v == 1]
    if active_flags:
        print(f"\n   ✨ Attributes: {', '.join(active_flags)}")

    if result.get('reference_used'):
        print(f"\n   📋 Reference: '{result['reference_used']}'  (ML raw: ${result['ml_raw_price']:.2f})")

    print(f"\n{SEPARATOR}\n")
    return result


# ═══════════════════════════════════════════════════════════════
# AI PRICING AGENT
# ═══════════════════════════════════════════════════════════════

def display_ai_result(result: dict, title: str):
    """Rich display for AI pricing recommendation."""
    persona = result.get("persona", "Agent")
    source = result.get("_source", "unknown")
    source_label = "🤖 AI + ML" if source == "llm" else "⚙️ ML Fallback"

    print_section(f"AI Pricing Recommendation  [{source_label}]", "💲")
    print(f"\n   📦 Product:   {title[:75]}{'...' if len(title) > 75 else ''}")
    print(f"   🎭 Persona:   {persona}")

    ml_data = result.get("ml_data", {})
    if ml_data:
        print(f"   🏷️  Brand:     {ml_data.get('brand', 'N/A')}")
        print(f"   📁 Category:  {ml_data.get('category', 'N/A')}")

    # Product validation (semantic filter)
    validation = result.get("product_validation", {})
    if validation:
        v_match = validation.get("is_match", False)
        v_type = validation.get("match_type", "unknown")
        v_conf = validation.get("confidence", 0)
        v_reason = validation.get("reason", "")

        if v_type == "accessory":
            print(f"\n   ⚠️  PRODUCT ALERT: Accessory detected! (confidence: {v_conf:.0%})")
            print(f"      This appears to be an accessory, not a main product.")
            if v_reason:
                print(f"      Reason: {v_reason[:70]}")
            print(f"      💡 The price recommendation may not be accurate.")
        elif v_type == "unrelated":
            print(f"\n   ⚠️  PRODUCT ALERT: Category mismatch! (confidence: {v_conf:.0%})")
            print(f"      This product may not belong to the selected category.")
            if v_reason:
                print(f"      Reason: {v_reason[:70]}")
        elif v_match:
            print(f"\n   ✅ Product verified: {v_type} match ({v_conf:.0%} confidence)")

    print(f"\n   ╔════════════════════════════════════════╗")
    print(f"   ║  💵  Suggested Price:  ${result['suggested_price']:.2f}")
    pr = result.get("price_range", {})
    print(f"   ║  📊  Price Range:      ${pr.get('min', 0):.2f} — ${pr.get('max', 0):.2f}")
    print(f"   ║  🎯  Confidence:       {result.get('confidence', 0):.0%}")
    print(f"   ║  📋  Strategy:         {result.get('strategy_name', 'N/A')}")
    print(f"   ╚════════════════════════════════════════╝")

    # ML comparison
    if ml_data:
        raw = ml_data.get("ml_raw_price", 0)
        adj = ml_data.get("ml_adjusted_price", 0)
        ref = ml_data.get("reference_used")
        print(f"\n   📈 ML Details:")
        print(f"      • Raw ML:     ${raw:.2f}")
        print(f"      • Adjusted:   ${adj:.2f}")
        if ref:
            print(f"      • Reference:  {ref}")

    # Reasoning
    reasoning = result.get("reasoning", "")
    if reasoning:
        print(f"\n   💡 Reasoning:")
        # Word-wrap
        words = reasoning.split()
        line = "      "
        for w in words:
            if len(line) + len(w) + 1 > 75:
                print(line)
                line = "      " + w
            else:
                line += " " + w if line.strip() else "      " + w
        if line.strip():
            print(line)

    # Pricing factors
    factors = result.get("pricing_factors", [])
    if factors:
        print(f"\n   📊 Pricing Factors:")
        for pf in factors:
            if isinstance(pf, dict):
                factor_name = pf.get("factor", "")
                impact = pf.get("impact", "")
                expl = pf.get("explanation", "")
                print(f"      • {factor_name}: {impact}")
                if expl:
                    print(f"        └─ {expl[:70]}")
            else:
                print(f"      • {pf}")

    # Seller tips
    tips = result.get("seller_tips", [])
    if tips:
        print(f"\n   🎓 Seller Tips:")
        for i, tip in enumerate(tips, 1):
            print(f"      {i}. {tip}")

    print(f"\n{SEPARATOR}\n")


def run_agent_pricing(title: str, category: str, persona_id: str, subcategory: str = None):
    """Run full AI pricing agent."""
    # Lazy import to avoid loading LLM client if not needed
    from agents.pricing_agent import PricingAgent

    persona = get_persona(persona_id)
    print(f"\n   ⏳ Initializing {persona.emoji} {persona.name} agent...")

    agent = PricingAgent(persona=persona)
    status = agent.get_status()
    mode = "AI + ML" if status["llm_available"] else "ML Only (LLM unavailable)"
    sf_mode = "✅" if status.get("semantic_filter") else "❌ (keyword fallback)"
    print(f"   ✅ Agent ready  [{mode}]")
    print(f"   🔍 Semantic Filter: {sf_mode}")

    if not status["llm_available"]:
        print(f"   ⚠️  Gemini API not configured — using ML fallback mode")
        print(f"      Set GEMINI_API_KEY in .env to enable AI mode")

    print(f"   ⏳ Analyzing: {title[:60]}...")
    result = agent.price(title, category, subcategory)

    display_ai_result(result, title)
    return result


# ═══════════════════════════════════════════════════════════════
# SEMANTIC FILTER
# ═══════════════════════════════════════════════════════════════

def run_semantic_filter():
    """Interactive semantic filter mode."""
    from agents.llm_client import GeminiClient
    from agents.semantic_filter import SemanticFilter

    print_section("Semantic Product Filter", "🔍")
    print("   Verify if listing titles match your search query.\n")

    try:
        llm = GeminiClient()
        sf = SemanticFilter(llm)
        print("   ✅ Gemini LLM connected\n")
    except Exception as e:
        print(f"   ❌ LLM not available: {e}")
        print(f"      Set GEMINI_API_KEY in .env\n")
        return

    query = input("   🔎 Search query: ").strip()
    if not query:
        return

    print("\n   Enter listing titles (one per line, empty line to finish):")
    listings = []
    while True:
        line = input("      > ").strip()
        if not line:
            break
        listings.append(line)

    if not listings:
        return

    print(f"\n   ⏳ Filtering {len(listings)} listings...\n")
    results = sf.filter_batch(query, listings)

    # Display results
    match_count = sum(1 for r in results if r.get("is_match"))
    print(f"   ✅ {match_count}/{len(results)} listings match '{query}':\n")

    for i, r in enumerate(results):
        title = r.get("title", listings[i] if i < len(listings) else "?")
        icon = "✅" if r.get("is_match") else "❌"
        match_type = r.get("match_type", "?")
        confidence = r.get("confidence", 0)
        reason = r.get("reason", "")

        print(f"   {icon} [{match_type:>10}] {confidence:.0%}  {title[:55]}")
        if reason:
            print(f"      └─ {reason[:70]}")

    print()


# ═══════════════════════════════════════════════════════════════
# PERSONA SELECTION
# ═══════════════════════════════════════════════════════════════

def select_persona() -> str:
    """Interactive persona selection."""
    personas = list_personas()
    print("\n   🎭 Choose your pricing persona:\n")
    for i, p in enumerate(personas, 1):
        print(f"      {i}. {p['name']}")
        print(f"         Goal:   {p['goal']}")
        print(f"         Margin: {p['margin']}")
        print(f"         Factor: {p['factor']}x")
        print()

    while True:
        choice = input("   Select (1-3): ").strip()
        if choice in ("1", "2", "3"):
            keys = list(PERSONAS.keys())
            return keys[int(choice) - 1]
        print("   ⚠️  Please enter 1, 2, or 3")


def select_category() -> str:
    """Interactive category selection."""
    print_categories()
    while True:
        choice = input("   Category number (or name): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(CATEGORIES):
                return CATEGORIES[idx]
            print(f"   ⚠️  Enter 1-{len(CATEGORIES)}")
        else:
            # Try to match by name
            for cat in CATEGORIES:
                if choice.lower() in cat.lower():
                    return cat
            print(f"   ⚠️  Category not found. Try a number.")


# ═══════════════════════════════════════════════════════════════
# INTERACTIVE MENU
# ═══════════════════════════════════════════════════════════════

def interactive_menu():
    """Main interactive menu loop."""
    print_banner()

    # Show LLM status
    llm_status = "✅ Configured" if Settings.is_llm_configured() else "❌ Not set (ML-only mode)"
    print(f"   Gemini API: {llm_status}")
    print(f"   Model:      {Settings.GEMINI_MODEL}")
    print(f"   Categories: {len(CATEGORIES)} available\n")

    while True:
        print(f"\n{'─' * 50}")
        print("   📋 MAIN MENU\n")
        print("      1. 💰  Quick ML Prediction")
        print("      2. 🤖  AI Pricing Agent (with Persona)")
        print("      3. 🔍  Semantic Product Filter")
        print("      4. 📊  Compare All Personas")
        print("      5. 📂  Show Categories")
        print("      6. ❌  Exit")
        print()

        choice = input("   Select (1-6): ").strip()

        if choice == "1":
            # Quick ML prediction
            print_section("Quick ML Prediction", "💰")
            title = input("\n   📦 Product title: ").strip()
            if not title:
                continue
            category = select_category()
            quick_predict(title, category)

        elif choice == "2":
            # AI Agent pricing
            print_section("AI Pricing Agent", "🤖")
            title = input("\n   📦 Product title: ").strip()
            if not title:
                continue
            category = select_category()
            persona_id = select_persona()
            run_agent_pricing(title, category, persona_id)

        elif choice == "3":
            # Semantic filter
            run_semantic_filter()

        elif choice == "4":
            # Compare all personas
            print_section("Compare All Personas", "📊")
            title = input("\n   📦 Product title: ").strip()
            if not title:
                continue
            category = select_category()
            compare_personas(title, category)

        elif choice == "5":
            print_categories()

        elif choice == "6":
            print("\n   👋 Au revoir!\n")
            break

        else:
            print("   ⚠️  Please enter 1-6")


# ═══════════════════════════════════════════════════════════════
# COMPARE ALL PERSONAS
# ═══════════════════════════════════════════════════════════════

def compare_personas(title: str, category: str, subcategory: str = None):
    """Run pricing with all 3 personas and compare side-by-side."""
    from agents.pricing_agent import PricingAgent

    results = {}
    for pid in PERSONAS:
        persona = get_persona(pid)
        print(f"\n   ⏳ Running {persona.emoji} {persona.name}...")
        agent = PricingAgent(persona=persona)
        results[pid] = agent.price(title, category, subcategory)

    # Side-by-side comparison
    print_section("Persona Comparison", "📊")
    print(f"\n   📦 {title[:65]}\n")

    header = f"   {'Persona':<25} {'Price':>10} {'Range':>22} {'Confidence':>12}"
    print(header)
    print(f"   {'─' * 72}")

    for pid, res in results.items():
        persona = get_persona(pid)
        name = f"{persona.emoji} {persona.name}"
        price = f"${res['suggested_price']:.2f}"
        pr = res.get("price_range", {})
        prange = f"${pr.get('min', 0):.2f} — ${pr.get('max', 0):.2f}"
        conf = f"{res.get('confidence', 0):.0%}"

        print(f"   {name:<25} {price:>10} {prange:>22} {conf:>12}")

    # Show differences
    prices = [r["suggested_price"] for r in results.values()]
    if prices:
        spread = max(prices) - min(prices)
        print(f"\n   📈 Price spread: ${spread:.2f} (${min(prices):.2f} — ${max(prices):.2f})")

    # ML baseline
    ml_data = list(results.values())[0].get("ml_data", {})
    if ml_data:
        print(f"   📉 ML baseline:  ${ml_data.get('ml_adjusted_price', 0):.2f}")
        if ml_data.get("reference_used"):
            print(f"   📋 Reference:    {ml_data['reference_used']}")

    print(f"\n{SEPARATOR}\n")


# ═══════════════════════════════════════════════════════════════
# CLI ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="PricePulse — AI-Powered E-Commerce Pricing Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                                          Interactive mode
  python app.py --quick "iPhone 15 Pro 256GB" "Electronics"
  python app.py --persona optimizer "MacBook Air M3" "Computers & Accessories"
  python app.py --compare "Samsung Galaxy S24" "Electronics"
        """,
    )

    parser.add_argument("--quick", nargs=2, metavar=("TITLE", "CATEGORY"),
                        help="Quick ML prediction (no LLM)")
    parser.add_argument("--persona", nargs=3, metavar=("PERSONA", "TITLE", "CATEGORY"),
                        help="AI pricing with persona (maximize_sales|maximize_profit|optimizer)")
    parser.add_argument("--compare", nargs=2, metavar=("TITLE", "CATEGORY"),
                        help="Compare all 3 personas side-by-side")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON instead of formatted display")

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.quick:
        title, category = args.quick
        if args.json:
            result = ml_predict(title, category)
            print(json.dumps(result, indent=2, default=str))
        else:
            quick_predict(title, category)

    elif args.persona:
        persona_id, title, category = args.persona
        if args.json:
            from agents.pricing_agent import PricingAgent
            agent = PricingAgent(persona=persona_id)
            result = agent.price(title, category)
            print(json.dumps(result, indent=2, default=str))
        else:
            run_agent_pricing(title, category, persona_id)

    elif args.compare:
        title, category = args.compare
        if args.json:
            from agents.pricing_agent import PricingAgent
            results = {}
            for pid in PERSONAS:
                agent = PricingAgent(persona=pid)
                results[pid] = agent.price(title, category)
            print(json.dumps(results, indent=2, default=str))
        else:
            compare_personas(title, category)

    else:
        interactive_menu()


if __name__ == "__main__":
    main()

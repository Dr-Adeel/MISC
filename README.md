# MISC
Distribution des travaux


## Development of microservices and distributed systems

Instead of a single, large "monolith," application will be built as a collection of small, independent services.
Ensure services can operate and scale independently. The single entry point that routes requests to the appropriate microservices.

Project : PricePulse AI Recommendation System

Develop an OOP-driven application where a seller inputs a product description/image. The system autonomously researches the current market (eBay, Amazon, etc.), analyzes the pricing trends, and uses an AI agent to recommend an optimal listing price based on specified business goals (e.g., "fast sale" vs. "maximum profit").

Module A: The Market Scrapers (Inheritance & Abstraction)
Abstract Class MarketProvider: Defines methods like fetch_listings(product_name) and clean_data().

Subclasses EbayProvider, AmazonProvider, WalmartProvider: Each implements its own logic for API calls or HTML parsing (Polymorphism).

Module B: The Analytical Engine (Encapsulation)
Class PricePoint: A data object representing a single listing (price, condition, shipping cost, seller rating).

Class MarketAnalyzer: Encapsulates the logic for calculating mean, median, and outliers. It should hide the complexity of the statistical math from the rest of the system.

Module C: The AI Pricing Agent (Composition)
Class PricingAgent: This class has a (Composition) Connection to an LLM. It takes the summarized market data and a "Seller Persona" (e.g., "Aggressive Seller") to generate a natural language justification for the recommended price.

Module D: Explainability



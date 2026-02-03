# MISC
Distribution des travaux


## Development of microservices and distributed systems

Instead of a single, large "monolith," application will be built as a collection of small, independent services.
Ensure services can operate and scale independently. The single entry point that routes requests to the appropriate microservices.

### Project : PricePulse AI Recommendation System

Develop an OOP-driven application where a seller inputs a product description/image. The system autonomously researches the current market (eBay, Amazon, etc.), analyzes the pricing trends, and uses an AI agent to recommend an optimal listing price based on specified business goals (e.g., "fast sale" vs. "maximum profit").

#### Module A: The Market Scrapers (Inheritance & Abstraction)

Abstract Class MarketProvider: Defines methods like fetch_listings(product_name) and clean_data().

Subclasses EbayProvider, AmazonProvider, WalmartProvider: Each implements its own logic for API calls or HTML parsing (Polymorphism).

#### Module B: The Analytical Engine (Encapsulation)

Class PricePoint: A data object representing a single listing (price, condition, shipping cost, seller rating).

Class MarketAnalyzer: Encapsulates the logic for calculating mean, median, and outliers. It should hide the complexity of the statistical math from the rest of the system.

#### Module C: The AI Pricing Agent (Composition)

Class PricingAgent: This class has a (Composition) Connection to an LLM. It takes the summarized market data and a "Seller Persona" (e.g., "Aggressive Seller") to generate a natural language justification for the recommended price.

#### Module D: Explainability 

The modules, help decides the distinguishing features for a recommended price.

- Phase 1	Interface Design: 	Define the Abstract Base Classes (ABCs) to ensure the integrated code will be compatible.

- Phase 2	Data Acquisition:	Implement the specific scrapers or API wrappers (Ebay/Amazon mocks).

- Phase 3	AI Integration:	Use an LLM to interpret "messy" product titles and determine if they actually match the seller's item.

- Phase 4	The Orchestrator:	A PriceController class that runs the workflow from input to final report.

- Phase 5	Testing:	Starting from unit tests, each service class must respect the tests.

- Phase 6	Explainability:	Add explanations to gain confidence on the recommended price.

##### Error Handling: Create custom Exception classes (e.g., MarketUnreachableError, InsufficientDataError) to handle the volatility of web data.

##### The Orchestrator: PriceController Specification
The PriceController follows the Mediator Pattern. It ensures that the Scraping team, the Math team, and the AI team can work independently.


#### Data Engineers (EL JMILI	Hamza, EL KHELYFY	Imad, JADDI	Imad) : Implementing EbayProvider, AmazonProvider, and handling MarketUnreachableError.

#### Data Scientists (AFFOUDJI	Akomédi Paterne, TAFOUGHALTI	Youssef, YOUNOUSSA SOUNA	Abdoul Wahab) : Building the MarketAnalyzer and statistical outlier logic.

#### AI/Prompt Engineers (BOUKECHOUCH	Mohamed, EL YOUSFI	Ali, Elhaddouchi	Maryam) : Tuning the PricingAgent prompts and "Persona" logic.

#### DevOps (SALIM	Ayoub, SENHAJI	Tarik) : Creating and Testing endpoints


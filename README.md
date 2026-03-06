# 🏷️ PricePulse

**AI-Powered Price Prediction Engine for Amazon Sellers**

PricePulse is an intelligent pricing engine that combines a **GradientBoosting ML model** with **Google Gemini LLM reasoning** and **seller persona strategies** to recommend optimal prices across 24 Amazon product categories. It features 363 reference prices, multi-language support, and an accessory-vs-device detection system.

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| **ML Prediction** | GradientBoostingRegressor trained on 50,444 Amazon products |
| **363 Reference Prices** | Calibrated against real market prices (iPhone, MacBook, PS5, Dyson, etc.) |
| **AI Agent Layer** | Google Gemini LLM (`gemini-2.0-flash`) for enhanced reasoning & market analysis |
| **3 Seller Personas** | Maximize Sales · Maximize Profit · Balanced Optimizer |
| **Semantic Filter** | Detects accessories vs. main products (cables, cases, controllers…) |
| **24 Categories** | Electronics, Clothing, Home, Sports, Toys, Beauty, Books, Office & more |
| **Spec Extraction** | Parses storage, RAM, screen size, resolution, pack qty from product titles |
| **Multi-Language** | Handles French, Spanish, German, Arabic product titles |
| **Explainability** | Confidence scores, price ranges, reasoning breakdown, seller tips |

---

## 📁 Project Structure

```
PricePulse/
├── app.py                       # CLI entry point (interactive menu + CLI args)
├── predict_price.py             # Core ML prediction engine (2,600+ lines)
├── retrain_model.py             # Model retraining script
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── .gitignore
│
├── agents/                      # AI Agent Layer
│   ├── pricing_agent.py         # Orchestrator: ML → LLM → Persona pipeline
│   ├── llm_client.py            # Google Gemini API client
│   └── semantic_filter.py       # Product vs accessory classifier
│
├── config/
│   ├── settings.py              # Global configuration (API keys, model params)
│   └── personas.py              # 3 seller persona definitions
│
├── prompts/
│   ├── pricing_prompts.py       # LLM prompt templates for pricing
│   └── filter_prompts.py        # Semantic filter prompt templates
│
├── models/                      # Trained model artifacts
│   ├── model_price_predictor.pkl    # GradientBoostingRegressor
│   ├── tfidf_vectorizer.pkl         # TF-IDF(5000) vectorizer
│   ├── le_category.pkl              # Category label encoder
│   ├── le_subcategory.pkl           # Subcategory label encoder
│   ├── le_brand.pkl                 # Brand label encoder
│   └── model_metadata.json         # Reference prices, brands, medians (735 lines)
│
├── data/
│   └── final_dataset_cleaned.csv   # 50,444 products × 24 categories
│
├── notebooks/
│   └── DataAnalysis.ipynb          # Exploratory data analysis
│
└── tests/
    ├── test_predictions.py         # 133 prediction tests (all 24 categories)
    ├── test_edge_cases.py          # 101 edge-case tests (accessories, multi-lang…)
    └── test_agent.py               # 9 agent integration tests
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd PricePulse

# Create environment (conda or venv)
conda create -n pp_ai python=3.12 -y
conda activate pp_ai

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key *(optional — ML works without it)*

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

### 3. Launch

```bash
python app.py
```

---

## 💡 Usage

### Interactive Menu

```bash
python app.py
```

```
╔═══════════════════════════════════════════════════════════╗
║    💲  P R I C E   P U L S E   v 2 . 0                  ║
║    AI-Powered E-Commerce Pricing Engine                  ║
║    ML Prediction  ·  Gemini AI  ·  Seller Personas       ║
╚═══════════════════════════════════════════════════════════╝

1. ⚡ Quick Price (ML-only)
2. 📦 Batch Predict
3. 🤖 AI Agent Mode (ML + Gemini + Persona)
4. 🔍 Semantic Filter Test
5. 🧪 Run Tests
```

### CLI Quick Mode

```bash
# ML-only quick prediction
python app.py --quick "iPhone 15 Pro 256GB" "Electronics - Mobile & Accessories"

# AI agent with persona
python app.py --persona optimizer "MacBook Air M3 256GB" "Computers & Accessories"
```

### Python API

```python
from predict_price import predict_price

result = predict_price("Sony WH-1000XM5 Headphones", "Electronics - Audio")
print(f"Price: ${result['predicted_price']:.2f}")
print(f"Range: ${result['price_range'][0]:.2f} – ${result['price_range'][1]:.2f}")
print(f"Brand: {result['brand_detected']}")
```

### AI Agent API

```python
from agents.pricing_agent import PricingAgent

agent = PricingAgent(persona="maximize_profit")
result = agent.price("Apple MacBook Air M3 256GB", "Computers & Accessories")
print(result)
```

---

## 🧪 Testing

PricePulse includes **243 total tests** across 3 test suites:

```bash
# 133 prediction tests — covers all 24 categories
python tests/test_predictions.py

# 101 edge-case tests — accessories, multi-language, ambiguous titles
python tests/test_edge_cases.py

# 9 agent integration tests
python tests/test_agent.py
```

### Edge Cases Covered

| Pattern | Examples |
|---------|----------|
| Console accessories priced as devices | PS5 controller, Xbox headset, Nintendo Joy-Con |
| Multi-language titles | French, Spanish, German, Arabic product names |
| Accessory vs. device misclassification | iPhone case, laptop charger, screen protector |
| Pack/count inflation | "Pack of 6 batteries", "3-piece cookware set" |
| Budget/unknown brands | Generic electronics, unbranded accessories |
| Spec extraction false positives | Numbers in titles that aren't specs |
| Streaming sticks vs. smart TVs | Fire TV Stick, Chromecast vs. 65" OLED |

---

## 🔧 Model Details

| Metric | Value |
|--------|-------|
| **Algorithm** | GradientBoostingRegressor |
| **Training Data** | 50,444 Amazon products |
| **R² Score** | 0.587 |
| **MAE** | $25.99 |
| **Features** | TF-IDF(5000) + Category + Brand + Medians + Specs + Flags |
| **Reference Prices** | 363 known products |
| **Known Brands** | 58 |

### Prediction Pipeline

```
Product Title + Category
         │
         ▼
┌─────────────────────────┐
│  1. Feature Extraction   │ ← TF-IDF, brand detection, spec parsing
├─────────────────────────┤
│  2. ML Prediction        │ ← GradientBoosting (log-price space)
├─────────────────────────┤
│  3. Reference Calibration│ ← 363 known product prices
├─────────────────────────┤
│  4. Spec Adjustments     │ ← Storage, RAM, screen size, resolution, pack qty
├─────────────────────────┤
│  5. Accessory Detection  │ ← Caps prices for cases, cables, chargers
├─────────────────────────┤
│  6. Category Guards      │ ← Per-category min/max price bounds
├─────────────────────────┤
│  7. Smart Blending       │ ← Weighted blend of ML + heuristic estimates
└─────────────────────────┘
         │
         ▼
  Final Price + Range + Confidence
```

### Categories Supported (24)

| Domain | Categories |
|--------|-----------|
| **Electronics** | Computers, Mobile & Accessories, Audio, TV & Video, Gaming, Cameras, Wearables |
| **Fashion** | Clothing & Accessories, Shoes, Watches, Jewelry |
| **Home** | Home & Kitchen, Tools & Home Improvement, Garden & Outdoor |
| **Personal** | Beauty & Personal Care, Health & Household |
| **Family** | Baby Products, Toys & Games |
| **Media** | Books, Musical Instruments |
| **Other** | Sports & Outdoors, Office Products, Pet Supplies, Automotive |

---

## 🤖 AI Agent Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       PricingAgent                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐    ┌──────────────┐   ┌──────────────┐  │
│  │ Semantic Filter │───▶│ ML Predict   │──▶│ LLM Enhance  │  │
│  │ (product valid?)│    │ (GradientBst)│   │ (Gemini API) │  │
│  └────────────────┘    └──────────────┘   └──────┬───────┘  │
│                                                   │          │
│                                          ┌────────▼───────┐  │
│                                          │ Persona Adjust │  │
│                                          │ (Sales/Profit/ │  │
│                                          │  Optimizer)    │  │
│                                          └────────────────┘  │
│                                                              │
│  Graceful fallback: ML-only when LLM unavailable            │
└──────────────────────────────────────────────────────────────┘
```

### Seller Personas

| Persona | Factor | Strategy |
|---------|--------|----------|
| 🚀 **Sales Maximizer** | ×0.88 | 12% below market — volume & market share |
| 💰 **Profit Maximizer** | ×1.15 | 15% premium — margin optimization |
| ⚖️ **Optimizer** | ×1.00 | Balanced — competitive positioning |

Each persona injects a custom system prompt into the Gemini LLM, shaping the reasoning style, seller tips, and final price recommendation.

---

## 🔑 Key Modules

### `predict_price.py` — ML Engine (2,600+ lines)

The core prediction engine handling:
- **Brand detection** with 58 known brands and alias mapping (e.g., "Galaxy" → Samsung)
- **Spec extraction** via regex: storage (GB/TB), RAM, screen size, resolution, megapixels
- **Reference price matching** against 363 calibrated product prices
- **Category-specific estimators** for gaming, cameras, clothing, health, office, etc.
- **Accessory classification** to prevent cases/cables from getting device-level prices
- **Smart blending** between ML predictions and heuristic estimates

### `agents/pricing_agent.py` — AI Orchestrator

Combines ML prediction with Gemini LLM reasoning:
1. Validates the product via semantic filter
2. Gets ML base price from `predict_price`
3. Builds a prompt with product context + ML result
4. Sends to Gemini for enhanced reasoning
5. Applies persona adjustments to final price

### `config/personas.py` — Seller Strategies

Each `Persona` dataclass defines: business goal, price factor, margin target, LLM system prompt, pricing rules, and seller tips style.

---

## ⚙️ Dependencies

```
scikit-learn >= 1.3.0    # ML model
numpy >= 1.24.0          # Numerical computing
pandas >= 2.0.0          # Data processing
scipy >= 1.11.0          # Sparse matrices
joblib >= 1.3.0          # Model serialization
google-genai >= 1.0.0    # Gemini LLM API
python-dotenv >= 1.0.0   # Environment variables
```

---

## 📜 License

MIT License

---

## 🙏 Acknowledgments

- **Dataset:** Amazon Product Dataset (Kaggle) — 50,444 products
- **LLM:** Google Gemini API (`gemini-2.0-flash`)
- **ML:** scikit-learn GradientBoostingRegressor

---

Made with ❤️ for Amazon Sellers

# Quality Testing Report — PricePulse AI Engine

**Author:** QA Engineer Chaimaa ALLALI  
**Scope:** Functional and logic validation of the AI pricing engine and its semantic reasoning layer  

---

## 1. Objective

The goal of this testing session is to validate the reliability, resilience, and logical accuracy of PricePulse’s AI-powered pricing system.

The focus is on:

- ML fallback behavior (when LLM is unavailable)  
- Persona-dependent price variation  
- Input validation (empty, invalid, extreme, or nonsensical data)  
- Semantic filter correctness (main product vs accessory detection)  

---

## 3. Test Environment

- **Module under test:**  
  `agents/pricing_agent.py`

- **Mocked dependencies:**
  - `google.genai` (Gemini API) mocked to remove external API dependency  
  - `predict_price` replaced with a fake ML predictor returning dummy values  

- **Language:** Python 3.12  
- **Framework:** Pytest  

---



## 4. Test Scenarios Overview

| Category | Purpose | Expected Behavior |
|----------|--------|------------------|
| AI Output Structure | Validate existence of all key fields (price, range, confidence, reasoning) | Response contains all required keys |
| Positive Price Guard | Ensure no negative or zero price values | All predicted prices > 0 |
| Extreme Value Handling | Avoid unrealistic prices due to faulty ML outputs | Price stays within safe bounds |
| Persona Sensitivity | Verify personas affect pricing | Sales ≠ Profit results |
| Fallback Behavior | Check ML-only mode | `_source = ml_fallback` |
| Accessory Detection | Detect accessories | match_type = accessory |
| False Accessory Detection | Avoid misclassification | match_type ≠ accessory |
| Main Product with Accessory Words | Avoid wrong classification with mixed titles | match_type ≠ accessory |
| Invalid Persona | Handle unknown persona | Raises KeyError |
| Empty / Gibberish Input | Handle meaningless input | Returns structured response |
| Invalid Category | Handle unknown categories | Returns non-null output |
| Price Range Logic | Ensure consistency | min ≤ price ≤ max |
## 5. Observation Summary

| Test | Status | Issue Description |
|------|--------|------------------|
| test_false_accessory_detection | ❌ Failed | "iPhone 13 with case included" misclassified as accessory |
| test_main_product_with_accessory_words | ❌ Failed | "Samsung Galaxy phone with charger" misclassified |
| All other tests | ✅ Passed | Functionality and structure validated |

---

## 6. Analysis of Failures

Both failures occur in the semantic filter module, which decides if a product is a main item or an accessory.

**Root cause**

The system relies on simple keyword matching: If one of these appears, the product is classified as an accessory.

---

**Effect**

This leads to incorrect classification when a product includes an accessory.

Example:
---

**Suggested improvements**

- Add logic to distinguish "contains accessory" vs "is accessory"  
- Introduce confidence-based decisions  
- Improve handling of complex product titles  

---

## 7. Conclusion

The AI pricing engine performs consistently across core tests and handles difficult inputs such as empty or random data.

However, semantic misclassification shows a limitation in how the system interprets product descriptions.

Improving this will increase pricing accuracy and system reliability.

---

## Overall Quality Verdict

The system works well in fallback mode and handles persona logic correctly.

However, the semantic filtering still needs improvement, especially when understanding product descriptions.

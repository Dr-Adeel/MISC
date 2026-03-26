🧪 1.Quality Testing Report — PricePulse AI Engine


Author: QA Engineer
Scope: Functional & logic validation of the AI pricing engine and its semantic reasoning layer

🎯2. Objective


The goal of this testing session is to validate the reliability, resilience, and logical accuracy of PricePulse’s AI-powered pricing system, focusing on:

-ML fallback behavior (when LLM is unavailable)
-Persona-dependent price variation
-Input validation (empty, invalid, extreme, or nonsensical data)
-Semantic filter correctness (main product vs. accessory detection)


🧩3. Test Environment


-Module under test: agents/pricing_agent.py
-Mocked dependencies:
      --google.genai (Gemini API) mocked to remove external API dependency
      --predict_price replaced with a fake ML predictor returning dummy values
-Language: Python 3.12
-Framework: Pytest


🧠 4.Test Scenarios Overview



|Category	            |          Purpose	                                  |Expected Behavior
|AI Output Structure    |Validate existence of all key fields (price, range..)     |Response dictionary contains all required keys
|Positive Price Guard   |ensure no negative or zero price values                   |	All predicted prices > 0
|Extreme Value Handling |Avoid unrealistic prices due to faulty ML outputs         |	AI limits prices within safe bounds
|Persona Sensitivity	|Verify different seller personas change suggested price   | Sales Maximizer ≠ Profit Maximizer results
|Fallback Behavior	|Check ML-only mode when LLM unavailable                   | _source field marked as ml_fallback
|Accessory Detection    |Identify items like “iPhone case” as accessories          | Match type set to accessory
|False Accessory Detection |	Ensure “iPhone 13 with case included” is not misclassified | Match type ≠ accessory (✅ expected)
|Main Product with Accessory Words |	Avoid misclassification when accessory word appears alongside a main product	| Match type ≠ accessory (❌ failed)
|Invalid Persona	                 | raceful error on unknown persona	                                              | Raises KeyError
|Empty or Gibberish Input	         | Handle meaningless or blank titles safely	                                    | Returns structured response with low confidence
|Invalid Category	                 | Resilient to unknown categories	                                              | Returns non-null output
|Price Range Logic	               | Ensure suggested price falls within range bounds	                              | min ≤ price ≤ max
⚠️ 5.Observation Summary
Test	                                  |Status	         |Issue Description
test_false_accessory_detection	        |❌ Failed	     |AI misclassified “iPhone 13 with case included” as accessory
test_main_product_with_accessory_words	|❌ Failed	     |“Samsung Galaxy phone with charger” incorrectly tagged as accessory
All other tests	                        |✅ Passed	     |Functionality and structure validated successfully

🔍 6.Analysis of Failures
Both failures occur within the semantic filter module, which determines whether a product is a main item or an accessory.

=>Root cause: The filter likely relies on simple keyword matching (e.g., "case", "charger") without context weighting.
=>Effect: Phrases that contain accessory words even alongside a valid product (e.g., "phone with charger") are flagged as accessories.
=>Suggested improvement:
         -Add contextual logic to distinguish “contains accessory” vs. “is accessory.”
         -Introduce confidence-based thresholding (reduce false positives).
         -Expand training or fine-tuned patterns for compound product titles.
🧾 7.Conclusion
 The AI pricing engine performs consistently across core logic tests and shows strong fault tolerance for invalid inputs (empty, gibberish, extreme values).
However, two semantic misclassifications reveal a need for improved natural language understanding in accessory detection.
Once fine-tuned, the system will achieve full reliability in differentiating between main devices and included accessories, which is crucial for accurate pricing and seller trust.

Overall Quality Verdict:
✅ Robust ML fallback and persona logic
⚙️ Minor semantic refinement needed for contextual filtering

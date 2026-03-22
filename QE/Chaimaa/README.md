Quality Engineering Report – Data Science Module

1. Objective

The objective of this testing phase was to evaluate the reliability and correctness of the MarketAnalyzer module. This component is responsible for transforming raw market data into statistical insights, which are later used by the AI system to generate pricing recommendations. Therefore, the quality of this module directly impacts the overall system.

The focus was not only on verifying that the system executes correctly, but also on assessing whether the results produced are logically valid under different conditions.

---

2. Testing Approach

2.1 Use of Synthetic Data

Synthetic data was created in order to control specific scenarios and test edge cases. This allowed the simulation of situations that may not naturally appear in the dataset, such as negative prices, empty inputs, or irrelevant products.

This approach made it possible to isolate and test specific behaviors of the system.

2.2 Use of Real Dataset

The provided dataset (final_dataset.csv) was used to evaluate how the system behaves under realistic conditions. This helped identify issues related to data quality and preprocessing that cannot be observed using controlled data alone.

Combining both approaches ensured a more complete evaluation of the system.

---

3. Functional Validation

Initial tests confirmed that the system performs correctly under normal conditions. The MarketAnalyzer is able to process product data, apply filtering logic, and compute statistical measures such as mean and median.

Outlier detection using the IQR method works as expected when the input data is valid, and the system correctly handles cases where there are not enough samples to perform a reliable estimation.

These results indicate that the core statistical logic of the module is correctly implemented.

---

4. Issues Identified

4.1 Handling of Invalid Price Values

One of the main issues observed is that the system does not filter invalid price values such as zero or negative numbers. These values are included in the statistical calculations, which can lead to incorrect outputs.

For example, when the dataset contains values such as 100, 0, and -50, the resulting estimated price may be equal to zero. This is clearly not a valid outcome in the context of product pricing.

This issue indicates that the system assumes that input data is already clean, which is not always the case in real-world scenarios.

---

4.2 Handling of Empty Query Input

Another issue concerns the handling of empty user input. When an empty query is provided, the system does not reject it. Instead, it proceeds with the analysis and matches all available products.

As a result, a price estimation is still returned, even though the input does not specify any meaningful criteria. This behavior is logically incorrect and suggests the absence of input validation.

---

4.3 Data Type Inconsistency in the Dataset

During testing with the real dataset, it was observed that some price values are stored as strings rather than numeric values. This leads to failures in statistical computations, particularly when using numerical libraries such as NumPy.

This issue highlights a missing preprocessing step in the data pipeline. The system should ensure that all price values are properly converted to numeric types before performing calculations.

---

4.4 Outlier Filtering Fallback Behavior

The system includes a fallback mechanism in cases where all values are considered outliers. Instead of rejecting the data, it proceeds by using the full dataset.

While this approach prevents the system from failing, it introduces a risk of producing unreliable results. In such cases, the system prioritizes continuity over accuracy.

This behavior is not necessarily incorrect, but it represents a design limitation that should be acknowledged.

---

5. Positive Observations

Despite the issues identified, several aspects of the system function correctly.

The filtering mechanism based on product type behaves as expected, particularly in cases where no matching products are found. In such situations, the system correctly returns a failure response.

Additionally, the statistical computations are accurate when the input data is valid, and the system demonstrates stable behavior under standard conditions.

These observations confirm that the underlying analytical logic is sound.

---

6. General Analysis

The main weakness of the module lies not in its computational logic, but in its lack of validation mechanisms.

The system appears to be designed with the assumption that both the input data and the user query are valid. However, this assumption does not hold in real-world environments, where data can be noisy, incomplete, or inconsistent.

As a result, the system performs well under ideal conditions but may produce incorrect or misleading results when faced with imperfect input.

---

7. Conclusion

The MarketAnalyzer module provides a solid foundation for statistical analysis and price estimation. However, its reliability is limited by the absence of proper input and data validation.

The main issues identified include the handling of invalid price values, the acceptance of empty queries, and inconsistencies in data types within the dataset.

These issues do not necessarily cause the system to fail, but they significantly affect the correctness of the results.

From a Quality Engineering perspective, the module can be considered functional, but not fully robust. Improvements in validation and preprocessing are required to ensure reliable performance in real-world scenarios.


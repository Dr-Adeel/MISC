# Quality Engineering Report: Data Science Module

---

## 1. Objective

The objective of this testing phase was to evaluate the reliability and correctness of the `MarketAnalyzer` module. This component transforms raw market data into statistical insights that are later used by the AI system to generate pricing recommendations.

Because of this, any issue in this module directly affects the final pricing results.

The goal was not only to verify that the system runs correctly, but also to check whether the results produced are logically valid under different conditions.

---

## 2. Testing Approach

### 2.1 Use of Synthetic Data

Synthetic data was created to control specific scenarios and test edge cases.

This made it possible to simulate situations that may not appear naturally in the dataset, such as negative prices, empty inputs, or very small datasets.

This approach helped isolate specific behaviors and understand how the system reacts in unusual situations.

---

### 2.2 Use of Real Dataset

The provided dataset (`final_dataset.csv`) was used to evaluate how the system behaves under realistic conditions.

This helped identify issues related to data quality and preprocessing, such as incorrect formats or unexpected values.

Using real data also confirmed whether the system produces consistent results when working with larger inputs.

---

## 3. Functional Validation

Initial tests confirmed that the system behaves correctly under normal conditions.

The `MarketAnalyzer` is able to process product data, apply filtering logic, and compute statistical measures such as mean and median.

The results are consistent when the input data is clean and valid.

---

## 4. Identified Limitations

### 4.1 Handling of Invalid Prices

The system does not properly filter out negative or zero prices.

These values are still included in calculations, which can lead to incorrect results such as a fair price equal to zero.

---

### 4.2 Empty or Weak Queries

When the user input is empty, the system still returns results instead of rejecting the request.

This means the system can generate a price even when no clear product is provided.

---

### 4.3 Dependence on Data Quality

The system assumes that the dataset is already clean.

If the data contains incorrect formats (for example, prices stored as text), the system may fail or produce incorrect calculations.

---

### 4.4 Limited Validation of Results

The system computes statistics but does not always verify if the results are meaningful.

For example, it may return a result even when the number of matching products is very low, making the estimation unreliable.

---

## 5. Conclusion

The `MarketAnalyzer` module performs well when working with clean and structured data.

It correctly computes statistical values and applies filtering methods such as IQR to remove extreme values.

However, several limitations were identified. The system does not validate inputs strictly, depends heavily on data quality, and may produce results even when the input is not meaningful.

Improving input validation and data preprocessing would significantly increase the reliability of this module.

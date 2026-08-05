from QE.Chaimaa.data_scientist.market_analyzer import MarketAnalyzer
import pandas as pd
# fake product
class FakeProduct:
    def __init__(self, name, price, category="test", product_type="main_product"):
        self.name = name
        self.price = price
        self.category = category
        self.product_type = product_type
        self.rating = 4.5
        self.reviews_count = 100


# fake dataset
def fake_data():
    return [
        FakeProduct("iphone 13", 100),
        FakeProduct("iphone 13", 110),
        FakeProduct("iphone 13", 120),
        FakeProduct("iphone 13", 5000),  # extreme value Mr.outlier
    ]


# TESTING basic run
def test_basic_run():
    data = fake_data()
    analyzer = MarketAnalyzer(data)

    result = analyzer.get_price_estimate("iphone")

    print(result)

    assert result["success"] == True


# TESTING if outlier removed
def test_outlier_removed():
    data = fake_data()
    analyzer = MarketAnalyzer(data)

    result = analyzer.get_price_estimate("iphone")

    assert result["fair_price"] < 1000


# TESTING if the structure is correct
def test_output_structure():
    data = fake_data()
    analyzer = MarketAnalyzer(data)

    result = analyzer.get_price_estimate("iphone")

    assert "fair_price" in result
    assert "mean_price" in result
    assert "min_price" in result
    assert "max_price" in result


# TESTING if it works if there's not enough data
def test_not_enough_data():
    data = [FakeProduct("iphone", 100)]
    analyzer = MarketAnalyzer(data)

    result = analyzer.get_price_estimate("iphone")

    assert result["success"] == False

# TESTING if the price is reasonable (between 100 and 200)
def test_price_reasonable():
    data = fake_data()
    analyzer = MarketAnalyzer(data)

    result = analyzer.get_price_estimate("iphone")

    assert result["fair_price"] >= 100
    assert result["fair_price"] <= 200

# TESTING if the matching works (at least 1 match)
def test_product_matching():
    data = fake_data()
    analyzer = MarketAnalyzer(data)

    result = analyzer.get_price_estimate("iphone")

    assert result["total_matches"] > 0

#now let's test with real data from a CSV file
#cs price is in string format with $ and , so we need to clean it before use
def clean_price(p):
    try:
        return float(str(p).replace("$", "").replace(",", ""))
    except:
        return 0
#all good now but this is going on the report as the Data Scientists did not ensure numeric price format
def test_with_real_data():
    df = pd.read_csv("final_dataset.csv")
    products = [FakeProduct(row["title"], clean_price(row["price"]), row["category"]) for index, row in df.iterrows()]
    analyzer = MarketAnalyzer(products)

    result = analyzer.get_price_estimate("laptop")

    print(result)

    assert result["success"] == True

#is the data actually clean emmmmm let's test if zero and negative prices are handled correctly (they should be ignored)
def test_for_zero_and_negative_prices():
    data = [
        FakeProduct("iphone", 100),
        FakeProduct("iphone", 0),
        FakeProduct("iphone", -50),
    ]

    analyzer = MarketAnalyzer(data)
    result = analyzer.get_price_estimate("iphone")

    assert result["fair_price"] > 0

#okay okay so they forgot to remove invalid prices (0 or negative) 
#what happens when everything is weird
def test_all_outliers():
    data = [
        FakeProduct("iphone", 1000),
        FakeProduct("iphone", 2000),
        FakeProduct("iphone", 3000),
    ]

    analyzer = MarketAnalyzer(data)
    result = analyzer.get_price_estimate("iphone")

    assert result["success"] == True
    assert result["fair_price"] >= 1000
#when all is weird the system falls back to using the full dataset instead of rejecting unreliable data
#what if the query is just weird characters that don't match anything
def test_weird_query():
    data = fake_data()
    analyzer = MarketAnalyzer(data)

    result = analyzer.get_price_estimate("!!!!")

    assert result["success"] == False

#what if the query is empty
def test_empty_query():
    data = fake_data()
    analyzer = MarketAnalyzer(data)

    result = analyzer.get_price_estimate("")
    print(result)

    assert result["success"] == False

#what if the product type filter is used but doesn't match anything
def test_product_type_filter_no_match():
    data = fake_data()
    analyzer = MarketAnalyzer(data)

    result = analyzer.get_price_estimate("iphone", product_type="non_existent_type")

    assert result["success"] == False
    assert result["matches_count"] == 0

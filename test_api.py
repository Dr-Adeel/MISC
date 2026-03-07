from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_price_endpoint():

    response = client.post(
        "/estimate",
        json={"product_name": "iphone 13"}
    )

    assert response.status_code == 200

def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "degraded", "unhealthy"]
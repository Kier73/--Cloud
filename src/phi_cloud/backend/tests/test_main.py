import pytest
from fastapi.testclient import TestClient
from unittest import mock

from src.phi_cloud.backend.main import app

client = TestClient(app)

@mock.patch("src.phi_cloud.backend.main.API_KEY", "test-key")
def test_compile_hologram_identity():
    """
    Tests the /compile-hologram endpoint with a simple identity truth table.
    """
    response = client.post(
        "/compile-hologram",
        json={"truth_table": [[1, 1], [0, 0]]},
        headers={"X-API-Key": "test-key"}
    )

    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/png'
    assert len(response.content) > 1000

def test_compile_hologram_invalid_key():
    """
    Tests that the endpoint fails with an invalid API key.
    The real API_KEY is 'my-secret-key' by default.
    """
    response = client.post(
        "/compile-hologram",
        json={"truth_table": [[1, 1], [0, 0]]},
        headers={"X-API-Key": "invalid-key"}
    )
    assert response.status_code == 401

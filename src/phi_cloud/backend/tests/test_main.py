import pytest
from fastapi.testclient import TestClient
from unittest import mock

from src.phi_cloud.backend.main import app

client = TestClient(app)

# Define a valid request body for a NAND gate
NAND_GATE_CONFIG = {
    "input_ports": [64, 192],
    "output_ports": [128],
    "truth_table": [
        [[0, 0], [1]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ],
    "compiler_config": {
        "epochs": 1 # Use a single epoch for faster testing
    }
}

@mock.patch("src.phi_cloud.backend.main.API_KEY", "test-key")
def test_compile_hologram_success():
    """
    Tests a successful compilation request, checking the JSON response structure.
    """
    response = client.post(
        "/compile-hologram",
        json=NAND_GATE_CONFIG,
        headers={"X-API-Key": "test-key"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "image_url" in data
    assert "verification" in data
    assert "loss_history" in data
    assert data["verification"]["total_cases"] == 4

def test_compile_hologram_invalid_key():
    """
    Tests that the endpoint fails with an invalid API key.
    """
    response = client.post(
        "/compile-hologram",
        json=NAND_GATE_CONFIG,
        headers={"X-API-Key": "invalid-key"}
    )
    assert response.status_code == 401

def test_compile_hologram_bad_request():
    """
    Tests that the endpoint returns a 422 Unprocessable Entity for malformed data.
    """
    bad_config = { "input_ports": [64] }
    response = client.post(
        "/compile-hologram",
        json=bad_config,
        headers={"X-API-Key": "test-key"}
    )
    assert response.status_code == 422

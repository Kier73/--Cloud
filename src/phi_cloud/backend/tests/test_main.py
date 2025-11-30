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
    ]
}

@mock.patch("src.phi_cloud.backend.main.API_KEY", "test-key")
def test_compile_hologram_nand_gate():
    """
    Tests the /compile-hologram endpoint with a 2-input, 1-output NAND gate.
    """
    response = client.post(
        "/compile-hologram",
        json=NAND_GATE_CONFIG,
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
    assert len(response.content) > 1000 # Check for a reasonably sized image
    assert len(response.content) > 1000

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
    bad_config = {
        "input_ports": [64],
        # Missing output_ports and truth_table
    }
    response = client.post(
        "/compile-hologram",
        json=bad_config,
        headers={"X-API-Key": "test-key"}
    )
    assert response.status_code == 422
    The real API_KEY is 'my-secret-key' by default.
    """
    response = client.post(
        "/compile-hologram",
        json={"truth_table": [[1, 1], [0, 0]]},
        headers={"X-API-Key": "invalid-key"}
    )
    assert response.status_code == 401

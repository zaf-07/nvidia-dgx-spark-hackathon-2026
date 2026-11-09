import os
from io import BytesIO

from PIL import Image

from fastapi.testclient import TestClient

os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("VISION_API_KEY", None)
os.environ.pop("VISION_API_URL", None)

from main import app  # noqa: E402

client = TestClient(app)


def test_health_does_not_expose_secrets():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["configured"] is False
    assert "key" not in response.text.lower()


def test_rejects_unsupported_media_type():
    response = client.post(
        "/v1/describe",
        headers={"X-Session-ID": "invalid-media-test"},
        files={"image": ("frame.txt", b"not-an-image", "text/plain")},
    )
    assert response.status_code == 415


def test_missing_provider_configuration_fails_closed():
    response = client.post(
        "/v1/describe",
        headers={"X-Session-ID": "missing-key-test"},
        files={"image": ("frame.jpg", b"fake-jpeg", "image/jpeg")},
        data={"task": "scene", "previous_scene_key": ""},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Vision service is not configured"


def test_tracking_endpoint_is_independent_of_groq_configuration():
    # A tiny valid PNG proves the low-latency endpoint does not require or call
    # the cloud narrator. YOLO is disabled in unit tests, so it returns an
    # explicit empty/disabled result rather than fabricating detections.
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color=(40, 60, 80)).save(buffer, format="PNG")
    image = buffer.getvalue()
    response = client.post(
        "/v1/track",
        headers={"X-Session-ID": "tracking-test"},
        files={"image": ("frame.png", image, "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["detector"] == "disabled"
    assert payload["objects"] == []
    assert payload["collision_warnings_enabled"] is False


def test_sign_sequence_requires_three_jpeg_frames():
    response = client.post(
        "/v1/interpret-sign",
        headers={"X-Session-ID": "invalid-sign-sequence-test"},
        json={"frames": ["data:image/jpeg;base64,ZmFrZQ=="] * 2},
    )
    assert response.status_code == 422


def test_sign_sequence_rejects_unbounded_transcript():
    response = client.post(
        "/v1/interpret-sign",
        headers={"X-Session-ID": "oversized-sign-transcript-test"},
        json={
            "frames": ["data:image/jpeg;base64,ZmFrZQ=="] * 3,
            "previous_transcript": "x" * 601,
        },
    )
    assert response.status_code == 422


def test_sign_sequence_fails_closed_without_provider_key():
    response = client.post(
        "/v1/interpret-sign",
        headers={"X-Session-ID": "missing-sign-key-test"},
        json={
            "frames": ["data:image/jpeg;base64,ZmFrZQ=="] * 3,
            "previous_transcript": "Hello",
        },
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Vision service is not configured"


def test_unisign_is_opt_in_and_does_not_load_from_health_check():
    status = client.get("/v1/unisign/status")
    assert status.status_code == 200
    payload = status.json()
    assert payload["enabled"] is False
    assert payload["loaded"] is False
    assert "checkpoint" not in payload


def test_unisign_experimental_endpoint_fails_closed_when_disabled():
    response = client.post("/v1/unisign/interpret", json={"raw_holistic": [{}], "experimental": True})
    assert response.status_code == 503
    assert "experimental" in response.json()["detail"].lower()


def test_healthz_alias_matches_health():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

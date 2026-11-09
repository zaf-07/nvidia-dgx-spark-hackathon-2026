"""Unit tests for Campus Memory & Shared Sight & Deaf-Blind Bridge."""

import os
import uuid
from fastapi.testclient import TestClient

os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("VISION_API_KEY", None)
os.environ.pop("VISION_API_URL", None)

from main import app  # noqa: E402

client = TestClient(app, headers={"X-Device-ID": "test-device", "X-Session-ID": "test-session"})
anonymous = TestClient(app)


def test_campus_endpoints_require_device_identity():
    assert anonymous.get("/v1/memory/alerts").status_code == 401
    assert anonymous.post("/v1/memory/resolve/1").status_code == 401
    assert anonymous.get("/v1/bridge/messages?session_id=x").status_code == 401


def test_get_campus_alerts_returns_only_real_reports():
    response = client.get("/v1/memory/alerts")
    assert response.status_code == 200
    alerts = response.json()
    assert isinstance(alerts, list)
    for item in alerts:
        assert "id" in item
        assert "location" in item
        assert "category" in item
        assert "description" in item
        assert "relative_time" in item


def test_filter_alerts_by_location():
    response = client.get("/v1/memory/alerts?location=Library")
    assert response.status_code == 200
    alerts = response.json()
    for a in alerts:
        assert "Library" in a["location"] or a["location"] == "All Campus"


def test_post_observation_and_resolve():
    new_obs = {
        "location": "Room 204",
        "category": "hazard",
        "label": "spilled coffee",
        "description": "Coffee spilled near desk 3.",
        "severity": "medium",
    }
    response = client.post(
        "/v1/memory/observe",
        headers={"X-Session-ID": "student-test-device"},
        json=new_obs,
    )
    assert response.status_code == 200
    alert = response.json()
    alert_id = alert["id"]
    assert alert["label"] == "spilled coffee"
    assert alert["reported_by"] == "student-test-device"

    # Resolve it
    res = client.post(f"/v1/memory/resolve/{alert_id}")
    assert res.status_code == 200
    assert res.json()["status"] == "resolved"


def test_what_changed_detects_delta():
    location = "integration-test-delta-room"
    baseline = [
        {"label": "table", "x": 0.2, "y": 0.5, "width": 0.6, "height": 0.4},
        {"label": "chair", "x": 0.25, "y": 0.55, "width": 0.15, "height": 0.25},
    ]
    first = client.post(
        "/v1/memory/what-changed",
        json={"location": location, "current_objects": baseline, "update_baseline": True},
    )
    assert first.status_code == 200
    current = [
        {"label": "table", "x": 0.2, "y": 0.5, "width": 0.6, "height": 0.4},
        {"label": "chair", "x": 0.25, "y": 0.55, "width": 0.15, "height": 0.25},
        {"label": "box", "x": 0.45, "y": 0.60, "width": 0.20, "height": 0.20},
    ]
    res = client.post(
        "/v1/memory/what-changed",
        json={"location": location, "current_objects": current, "update_baseline": False},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["has_changed"] is True
    assert "box" in body["added_objects"]
    assert "box" in body["speech"].lower()


def test_bridge_messages_lifecycle():
    session_id = f"test-bridge-session-{uuid.uuid4().hex}"

    # 1. Deaf signs "THANK YOU"
    r1 = client.post(
        "/v1/bridge/message",
        json={
            "session_id": session_id,
            "sender_role": "deaf",
            "text": "THANK YOU",
        },
    )
    assert r1.status_code == 200
    assert r1.json()["text"] == "THANK YOU"
    assert r1.json()["sender_role"] == "deaf"

    # 2. Blind replies "You're welcome"
    r2 = client.post(
        "/v1/bridge/message",
        json={
            "session_id": session_id,
            "sender_role": "blind",
            "text": "You're welcome, let me know if you need anything else.",
        },
    )
    assert r2.status_code == 200
    assert r2.json()["sender_role"] == "blind"

    # 3. Retrieve messages
    r3 = client.get(f"/v1/bridge/messages?session_id={session_id}")
    assert r3.status_code == 200
    msgs = r3.json()
    assert len(msgs) == 2
    assert msgs[0]["text"] == "THANK YOU"
    assert msgs[1]["sender_role"] == "blind"

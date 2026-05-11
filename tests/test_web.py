from pathlib import Path

import pytest

try:
    from fastapi.testclient import TestClient
except RuntimeError:
    TestClient = None

from rgsn.store import CandidateStore
from rgsn.web import WebSession, create_app


FIXTURE = Path(__file__).parent / "fixtures" / "tiny_words.vec"


def test_web_session_observes_and_suggests() -> None:
    store = CandidateStore.from_text_file(FIXTURE)
    session = WebSession.from_store(store)

    state = session.observe("water", 3)

    assert state["observation_count"] == 1
    assert state["best"]["candidate_id"] == "water"
    assert state["suggestions"]


@pytest.mark.skipif(TestClient is None, reason="FastAPI endpoint tests need httpx")
def test_fastapi_app_exposes_state_and_observe_endpoints() -> None:
    store = CandidateStore.from_text_file(FIXTURE)
    client = TestClient(create_app(store))

    initial = client.get("/api/state").json()
    observed = client.post("/api/observe", json={"word": "water", "rank": 3}).json()

    assert initial["candidate_count"] == 12
    assert observed["observation_count"] == 1
    assert observed["best"]["candidate_id"] == "water"


@pytest.mark.skipif(TestClient is None, reason="FastAPI endpoint tests need httpx")
def test_fastapi_app_runs_simulation() -> None:
    store = CandidateStore.from_text_file(FIXTURE)
    client = TestClient(create_app(store))

    response = client.post(
        "/api/simulate",
        json={"target_word": "river", "budget": 4, "seed_words": "road, tree, water"},
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["target_id"] == "river"
    assert payload["guesses"]

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
import uvicorn

from rgsn.contexto import ContextoSolver
from rgsn.dictionary import WordDictionary
from rgsn.oracle import SimilarityRankOracle
from rgsn.solver import WeakFeedbackSolver
from rgsn.store import CandidateStore
from rgsn.types import FeedbackObservation, ScoredCandidate


@dataclass(slots=True)
class WebSession:
    store: CandidateStore
    solver: ContextoSolver
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_store(cls, store: CandidateStore, *, metadata: dict[str, Any] | None = None) -> WebSession:
        return cls(store=store, solver=ContextoSolver(store), metadata=metadata or {})

    def reset(self) -> dict[str, Any]:
        self.solver = ContextoSolver(self.store)
        return self.state()

    def observe(self, word: str, rank: float) -> dict[str, Any]:
        self.solver.observe(word.strip().lower(), rank)
        return self.state()

    def state(self, *, suggestion_count: int = 10) -> dict[str, Any]:
        observations = self.solver.solver.observations
        best = self.solver.solver.best_observation()
        return {
            "candidate_count": len(self.store.candidates),
            "metadata": dict(self.metadata),
            "observation_count": len(observations),
            "constraint_count": len(self.solver.solver.machine.constraints()),
            "has_direction": self.solver.solver.machine.direction() is not None,
            "best": _observation_to_dict(best) if best is not None else None,
            "observations": [_observation_to_dict(item) for item in observations],
            "suggestions": [_scored_to_dict(item) for item in self.solver.suggest(k=suggestion_count)],
        }

    def simulate(
        self,
        *,
        target_word: str,
        budget: int = 25,
        stop_rank: int = 1,
        seed_words: list[str] | None = None,
    ) -> dict[str, Any]:
        target_id = target_word.strip().lower()
        oracle = SimilarityRankOracle(self.store, target_id)
        solver = WeakFeedbackSolver(self.store)
        seeds = [word.strip().lower() for word in seed_words or [] if word.strip()]
        trace = solver.simulate(oracle, budget=budget, stop_rank=stop_rank, seed_ids=seeds)
        return {
            "target_id": trace.target_id,
            "budget": budget,
            "stop_rank": stop_rank,
            "success_step": trace.success_step,
            "best_rank_history": list(trace.best_rank_history),
            "guesses": [_observation_to_dict(item) for item in trace.guesses],
            "top": oracle.top(k=min(10, len(self.store.candidates))),
        }


def run_server(
    *,
    store: CandidateStore,
    host: str = "127.0.0.1",
    port: int = 8765,
    metadata: dict[str, Any] | None = None,
) -> None:
    app = create_app(store, metadata=metadata)
    print(f"RGSN web UI listening at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


def create_app(store: CandidateStore, *, metadata: dict[str, Any] | None = None) -> FastAPI:
    app = FastAPI(title="RGSN Contexto Lab")
    session = WebSession.from_store(store, metadata=metadata)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/state")
    def state(k: int = 10) -> dict[str, Any]:
        return session.state(suggestion_count=k)

    @app.post("/api/observe")
    async def observe(request: Request) -> dict[str, Any]:
        try:
            payload = await _request_payload(request)
            return session.observe(word=str(payload["word"]), rank=float(payload["rank"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/reset")
    def reset() -> dict[str, Any]:
        return session.reset()

    @app.post("/api/simulate")
    async def simulate(request: Request) -> dict[str, Any]:
        try:
            payload = await _request_payload(request)
            return session.simulate(
                target_word=str(payload["target_word"]),
                budget=int(payload.get("budget", 25)),
                stop_rank=int(payload.get("stop_rank", 1)),
                seed_words=_split_seed_words(str(payload.get("seed_words", ""))),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the local RGSN Contexto web UI.")
    parser.add_argument("--embeddings", required=True, type=Path, help="Path to a text embedding file.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--max-words", default=None, type=int)
    parser.add_argument("--dictionary", default=None, type=Path, help="Optional allowed-word dictionary file.")
    parser.add_argument("--min-word-length", default=2, type=int)
    parser.add_argument("--max-word-length", default=None, type=int)
    args = parser.parse_args(argv)

    metadata: dict[str, Any] = {
        "embeddings_path": str(args.embeddings),
        "dictionary_path": str(args.dictionary) if args.dictionary is not None else None,
    }
    if args.dictionary is not None:
        dictionary = WordDictionary.from_text_file(
            args.dictionary,
            min_length=args.min_word_length,
            max_length=args.max_word_length,
        )
        store = CandidateStore.from_text_file(
            args.embeddings,
            lowercase_ids=True,
            max_items=args.max_words,
            allowed_ids=dictionary.words,
        )
        metadata["dictionary_size"] = len(dictionary)
        metadata["dictionary_overlap"] = len(store.candidates)
        metadata["dictionary_coverage"] = len(store.candidates) / len(dictionary) if len(dictionary) else 0.0
    else:
        store = CandidateStore.from_text_file(args.embeddings, lowercase_ids=True, max_items=args.max_words)
    run_server(store=store, host=args.host, port=args.port, metadata=metadata)


async def _request_payload(request: Request) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    return payload


def _split_seed_words(value: str) -> list[str]:
    return [item.strip().lower() for item in value.replace("\n", ",").split(",") if item.strip()]


def _observation_to_dict(observation: FeedbackObservation | None) -> dict[str, Any] | None:
    if observation is None:
        return None
    return {
        "candidate_id": observation.candidate_id,
        "rank": observation.rank,
        "metadata": dict(observation.metadata),
    }


def _scored_to_dict(scored: ScoredCandidate) -> dict[str, Any]:
    return {
        "candidate_id": scored.candidate.id,
        "score": scored.score,
        "components": dict(scored.components),
        "metadata": dict(scored.candidate.metadata),
    }


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RGSN Contexto Lab</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d7dde5;
      --accent: #0f766e;
      --accent-strong: #115e59;
      --warn: #b45309;
      --shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 28px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1 {
      font-size: 18px;
      margin: 0;
      font-weight: 700;
      letter-spacing: 0;
    }
    main {
      display: grid;
      grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
      gap: 18px;
      padding: 18px;
      max-width: 1320px;
      margin: 0 auto;
    }
    section, aside {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    aside {
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      align-self: start;
    }
    section { overflow: hidden; }
    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      min-height: 54px;
    }
    h2 {
      font-size: 15px;
      margin: 0;
      font-weight: 700;
      letter-spacing: 0;
    }
    form {
      display: grid;
      gap: 10px;
    }
    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    input {
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      font: inherit;
      color: var(--ink);
      background: #ffffff;
    }
    input:focus {
      outline: 2px solid rgba(15, 118, 110, 0.18);
      border-color: var(--accent);
    }
    button {
      height: 38px;
      border: 1px solid var(--accent);
      border-radius: 6px;
      padding: 0 12px;
      background: var(--accent);
      color: #ffffff;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--accent-strong); }
    button.secondary {
      color: var(--accent-strong);
      background: #ffffff;
    }
    button.secondary:hover { background: #edf7f5; }
    .buttons {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      min-height: 64px;
      background: #fbfcfd;
    }
    .stat b {
      display: block;
      font-size: 20px;
      line-height: 1.1;
    }
    .stat span {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 13px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    th {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      background: #fbfcfd;
    }
    .content-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(280px, 0.9fr);
      gap: 0;
    }
    .subpanel + .subpanel { border-left: 1px solid var(--line); }
    .empty {
      padding: 22px 16px;
      color: var(--muted);
      font-size: 13px;
    }
    .error {
      min-height: 18px;
      color: var(--warn);
      font-size: 12px;
      font-weight: 700;
    }
    .sim-output {
      padding: 0 16px 16px;
      display: grid;
      gap: 10px;
    }
    .source-note {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      font-weight: 700;
      background: #fbfcfd;
    }
    @media (max-width: 880px) {
      header { padding: 0 16px; }
      main { grid-template-columns: 1fr; padding: 12px; }
      .content-grid { grid-template-columns: 1fr; }
      .subpanel + .subpanel { border-left: 0; border-top: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <header>
    <h1>RGSN Contexto Lab</h1>
    <button class="secondary" id="refresh">Refresh</button>
  </header>
  <main>
    <aside>
      <div class="stats">
        <div class="stat"><b id="candidate-count">0</b><span>Candidates</span></div>
        <div class="stat"><b id="observation-count">0</b><span>Observed</span></div>
        <div class="stat"><b id="constraint-count">0</b><span>Constraints</span></div>
        <div class="stat"><b id="best-rank">-</b><span>Best rank</span></div>
      </div>
      <div class="source-note" id="source-note"></div>

      <form id="observe-form">
        <h2>Feedback</h2>
        <label>Word<input id="word" autocomplete="off" placeholder="water" required /></label>
        <label>Rank<input id="rank" type="number" min="1" step="1" placeholder="42" required /></label>
        <div class="buttons">
          <button type="submit">Observe</button>
          <button type="button" class="secondary" id="reset">Reset</button>
        </div>
        <div class="error" id="observe-error"></div>
      </form>

      <form id="simulate-form">
        <h2>Simulation</h2>
        <label>Target<input id="target-word" autocomplete="off" placeholder="river" required /></label>
        <label>Budget<input id="budget" type="number" min="1" step="1" value="10" /></label>
        <label>Stop rank<input id="stop-rank" type="number" min="1" step="1" value="1" /></label>
        <label>Seeds<input id="seed-words" autocomplete="off" placeholder="road, tree, water" /></label>
        <button type="submit">Run Simulation</button>
        <div class="error" id="simulate-error"></div>
      </form>
    </aside>

    <section>
      <div class="panel-header">
        <h2>Search Session</h2>
      </div>
      <div class="content-grid">
        <div class="subpanel">
          <div class="panel-header"><h2>Suggestions</h2></div>
          <table>
            <thead>
              <tr><th>Word</th><th>Score</th><th>Direction</th><th>Anchor</th><th>Redundancy</th></tr>
            </thead>
            <tbody id="suggestions"></tbody>
          </table>
          <div class="empty" id="suggestions-empty">No suggestions yet.</div>
        </div>
        <div class="subpanel">
          <div class="panel-header"><h2>History</h2></div>
          <table>
            <thead><tr><th>Word</th><th>Rank</th></tr></thead>
            <tbody id="history"></tbody>
          </table>
          <div class="empty" id="history-empty">No observations yet.</div>
        </div>
      </div>
      <div class="subpanel">
        <div class="panel-header"><h2>Simulation Result</h2></div>
        <div class="sim-output" id="simulation-result">
          <div class="empty">No simulation run yet.</div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const fmt = (value) => Number(value).toFixed(3);

    async function request(path, options = {}) {
      const response = await fetch(path, {
        headers: { "content-type": "application/json" },
        ...options,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || payload.detail || "Request failed");
      return payload;
    }

    async function loadState() {
      const state = await request("/api/state?k=12");
      renderState(state);
    }

    function renderState(state) {
      $("candidate-count").textContent = state.candidate_count;
      $("observation-count").textContent = state.observation_count;
      $("constraint-count").textContent = state.constraint_count;
      $("best-rank").textContent = state.best ? Number(state.best.rank).toFixed(0) : "-";
      renderSource(state.metadata || {});

      const suggestions = $("suggestions");
      suggestions.innerHTML = "";
      state.suggestions.forEach((item) => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td title="${item.candidate_id}">${item.candidate_id}</td>
          <td>${fmt(item.score)}</td>
          <td>${fmt(item.components.direction)}</td>
          <td>${fmt(item.components.best_anchor)}</td>
          <td>${fmt(item.components.redundancy)}</td>
        `;
        suggestions.appendChild(row);
      });
      $("suggestions-empty").style.display = state.suggestions.length ? "none" : "block";

      const history = $("history");
      history.innerHTML = "";
      [...state.observations].reverse().forEach((item) => {
        const row = document.createElement("tr");
        row.innerHTML = `<td title="${item.candidate_id}">${item.candidate_id}</td><td>${Number(item.rank).toFixed(0)}</td>`;
        history.appendChild(row);
      });
      $("history-empty").style.display = state.observations.length ? "none" : "block";
    }

    function renderSource(metadata) {
      const parts = [];
      if (metadata.embeddings_path) parts.push(`Embeddings: ${metadata.embeddings_path}`);
      if (metadata.dictionary_path) {
        const overlap = metadata.dictionary_overlap ?? "-";
        const size = metadata.dictionary_size ?? "-";
        parts.push(`Dictionary: ${overlap}/${size} words loaded`);
      }
      $("source-note").textContent = parts.join(" · ");
    }

    function renderSimulation(result) {
      const root = $("simulation-result");
      const best = result.best_rank_history.length ? result.best_rank_history[result.best_rank_history.length - 1] : "-";
      root.innerHTML = `
        <div class="chips">
          <span class="chip">Target: ${result.target_id}</span>
          <span class="chip">Best: ${best}</span>
          <span class="chip">Success: ${result.success_step || "-"}</span>
        </div>
        <table>
          <thead><tr><th>Step</th><th>Word</th><th>Rank</th></tr></thead>
          <tbody>
            ${result.guesses.map((item, index) => `<tr><td>${index + 1}</td><td>${item.candidate_id}</td><td>${Number(item.rank).toFixed(0)}</td></tr>`).join("")}
          </tbody>
        </table>
      `;
    }

    $("observe-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      $("observe-error").textContent = "";
      try {
        const state = await request("/api/observe", {
          method: "POST",
          body: JSON.stringify({ word: $("word").value, rank: $("rank").value }),
        });
        $("word").value = "";
        $("rank").value = "";
        renderState(state);
      } catch (error) {
        $("observe-error").textContent = error.message;
      }
    });

    $("simulate-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      $("simulate-error").textContent = "";
      try {
        const result = await request("/api/simulate", {
          method: "POST",
          body: JSON.stringify({
            target_word: $("target-word").value,
            budget: $("budget").value,
            stop_rank: $("stop-rank").value,
            seed_words: $("seed-words").value,
          }),
        });
        renderSimulation(result);
      } catch (error) {
        $("simulate-error").textContent = error.message;
      }
    });

    $("reset").addEventListener("click", async () => renderState(await request("/api/reset", { method: "POST", body: "{}" })));
    $("refresh").addEventListener("click", loadState);
    loadState();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

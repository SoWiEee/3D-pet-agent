# Plan — §14.6.4 LLM: local Ollama + multi-turn clarification + LLM-assisted grounding

**Branch:** `feat/llm-ollama`  **Spec:** §14.6.4  **Date:** 2026-06-12

**Goal:** Deepen the command-language layer with a real local LLM. Three sub-features,
all behind env flags, all with a deterministic fallback so the demo never blocks:

1. **Ollama parser backend.** A backend abstraction under the existing
   `PET_AGENT_LLM_PARSER=on` seam: `PET_AGENT_LLM_BACKEND ∈ {anthropic, ollama}`.
   The Ollama path drives a local model (`qwen2.5-coder:7b`, already pulled) via the
   official `ollama` Python client's structured-output `format=<JSON schema>` param,
   then validates through the same `CommandIntent` Pydantic gate. Falls back to the
   rule parser on any failure.
2. **Multi-turn clarification.** When grounding returns `clarification`, generate a
   *discriminating* question (LLM, else a templated fallback over the candidates).
   The user's reply — carrying a `session_id` on `/command` — is re-parsed with the
   prior turn as context and re-grounded. Server holds a small per-session dialogue
   state.
3. **LLM-assisted grounding.** When the heuristic `GroundingResolver` is low-confidence
   (`clarification` / `no_match` with a near-tie), the scene graph (objects + relations
   as JSON) + the utterance go to the LLM, which picks the target `object_id` and writes
   the justification into `NavigationGoal.explanation`. Gated by `PET_AGENT_LLM_GROUNDING=on`.

**Tech stack:** Python 3.12, `ollama` PyPI client (lazy-imported `.[llm]` extra),
Pydantic `CommandIntent`/`NavigationGoal`, FastAPI. Local model `qwen2.5-coder:7b`
served at `http://localhost:11434`.

**Non-negotiable design rules (from CLAUDE.md / spec):**
- LLM is **event-driven**, never per-frame. These calls fire only on `/command`.
- Every LLM path has a **deterministic fallback** (rule parser / templated question /
  heuristic resolver). A missing `ollama` package, an unreachable server, a schema
  failure, or a timeout must never break `/command`.
- Reuse existing contracts (`CommandIntent`, `NavigationGoal`, `GroundingResult`); do
  not invent new schemas where these fit. `NavigationGoal.explanation` already exists.
- New deps via `uv`; lint/format with `ruff`. Files ≤ 800 lines, functions < 50.

---

## Current state (verified)

- `src/language/llm_parser.py` (175 lines): `LLMCommandParser` — Anthropic forced
  tool-use, lazy SDK import, `client_factory` injection, returns `CommandIntent | None`,
  swallows all errors → None.
- `src/language/command_parser.py`: `parse_command(text)` honours `PET_AGENT_LLM_PARSER=on`
  via `_get_llm_parser()` → tries LLM, falls back to `RuleCommandParser`.
- `src/planning/grounding_resolver.py` (454 lines): `GroundingResolver.resolve(intent,
  map, graph) → GroundingResult` with `status ∈ {success, clarification, no_match,
  empty_map}`, `candidates: list[(track_id, score)]`, `ambiguity_margin=0.12`.
- `src/runtime/websocket_server.py` `/command` (line ~668): `CommandRequest(text: str)`
  → `parse_command` → `grounding_resolver.resolve` → branches per status, emits
  `runtime.ask(...)` + plans on success. **No `session_id`, no dialogue state today.**
- `src/config.py` `AppSettings`: `PET_AGENT_` prefix; add Ollama knobs here.

---

## Task 1 — Ollama parser backend (`PET_AGENT_LLM_BACKEND=ollama`)

**Files:** `pyproject.toml`, `src/language/llm_parser.py`, `src/language/command_parser.py`,
`src/config.py`, `tests/test_ollama_parser.py`.

- [ ] **Step 1 (dep):** Add `llm = ["ollama>=0.4"]` extra to `pyproject.toml`. Install
  `uv pip install -e ".[llm]"`. Verify `python -c "import ollama"`.
- [ ] **Step 2 (config):** `AppSettings` gains `ollama_model: str = "qwen2.5-coder:7b"`,
  `ollama_host: str = "http://localhost:11434"`. (`PET_AGENT_LLM_BACKEND` /
  `PET_AGENT_LLM_PARSER` stay env-only, read in the factory.)
- [ ] **Step 3 (TDD, RED):** `tests/test_ollama_parser.py`:
  - `OllamaCommandParser.parse("go to the red cup")` with an injected fake client that
    returns a JSON `CommandIntent` → returns a validated `CommandIntent`, `raw_text`
    forced to the input.
  - Malformed JSON / schema-invalid content → `None`.
  - `client_factory` raising (package/host missing) → `None` (no exception escapes).
  - `make_llm_parser()` returns `OllamaCommandParser` when `PET_AGENT_LLM_BACKEND=ollama`,
    `LLMCommandParser` otherwise.
- [ ] **Step 4 (GREEN):** Add `OllamaCommandParser` to `llm_parser.py`. Structured output:
  ```python
  resp = client.chat(
      model=self.model,
      messages=[{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}],
      format=CommandIntent.model_json_schema(),   # Ollama structured output
      options={"temperature": 0.0},
  )
  data = json.loads(resp["message"]["content"])   # also handle obj-with-attrs shape
  data["raw_text"] = text
  return CommandIntent(**data)
  ```
  Lazy `import ollama`; `client_factory` default builds `ollama.Client(host=...)`.
  All exceptions → log + `None`. Add `make_llm_parser()` factory reading
  `PET_AGENT_LLM_BACKEND` (default `anthropic`), and point `command_parser._get_llm_parser()`
  at it. Reuse the existing `SYSTEM_PROMPT` (drop the tool-use sentence for the JSON path).
- [ ] **Step 5 (live smoke, opt-in):** A test that skips unless Ollama is reachable
  (`pytest.importorskip("ollama")` + a `requests`/`httpx`-free socket probe to the host)
  parses one real utterance end-to-end. Marked `@pytest.mark.live` so the default suite
  stays hermetic.
- [ ] **Step 6:** `ruff check/format`, run `tests/test_ollama_parser.py`, commit
  `feat(llm): Ollama structured-output parser backend + make_llm_parser factory`.

---

## Task 2 — Multi-turn clarification (per-session dialogue state)

**Files:** `src/language/dialogue.py` (new), `src/runtime/websocket_server.py`,
`tests/test_dialogue.py`.

- [ ] **Step 1 (TDD, RED):** `tests/test_dialogue.py`:
  - `DialogueStore.open_clarification(session_id, intent, candidates, question)` then
    `get(session_id)` returns the pending turn; `resolve(session_id)` clears it.
  - `discriminating_question(candidates, objects)` returns a question that names the
    distinguishing attributes/relations (LLM path stubbed; templated fallback asserted
    deterministically, e.g. "Which box — the one on the left or the one near the cup?").
  - Bounded store: > N sessions evicts oldest (no unbounded growth).
- [ ] **Step 2 (GREEN):** `dialogue.py`:
  - `@dataclass PendingClarification(intent, candidates, question, created_at)`.
  - `DialogueStore` — capped dict keyed by `session_id` (LRU-ish eviction, e.g. 64).
  - `discriminating_question(candidates, semantic_map, *, llm=None)` — if an LLM parser/
    client is available, ask it for a short disambiguating question over the candidate
    objects' classes + attributes + relations; else a templated fallback enumerating the
    candidates' distinguishing features. Never raises.
  - `merge_followup(prior_intent, reply_text)` — re-parse `reply_text`; if it only carries
    a discriminator ("the red one", "left"), fold it into `prior_intent.target.attributes`
    / a relation, keeping `intent_type`. (Rule-first; LLM optional via existing parser.)
- [ ] **Step 3 (wire):** `CommandRequest` gains `session_id: str | None = None`. In
  `/command`:
  - If a pending clarification exists for `session_id`, treat this turn as the reply:
    `merge_followup` → re-ground → on success clear the pending state and proceed; on
    repeat ambiguity, ask again (bounded retries → fall through to a plain pick).
  - When grounding returns `clarification` and `session_id` is present, store the pending
    turn and `runtime.ask(discriminating_question(...))` instead of the canned string.
  - No `session_id` → today's behaviour exactly (canned clarification string).
- [ ] **Step 4:** `ruff`, run `tests/test_dialogue.py` + `tests/test_websocket*`, commit
  `feat(llm): multi-turn clarification with per-session dialogue state`.

---

## Task 3 — LLM-assisted grounding (`PET_AGENT_LLM_GROUNDING=on`)

**Files:** `src/planning/llm_grounding.py` (new), `src/runtime/websocket_server.py`,
`tests/test_llm_grounding.py`.

- [ ] **Step 1 (TDD, RED):** `tests/test_llm_grounding.py`:
  - `llm_pick_target(utterance, scene_graph, candidates, *, client) → (object_id, justification)`
    with a fake client returning a chosen id + reason → returns that pick; object_id not in
    candidates → `None` (no hallucinated targets).
  - Client/parse failure → `None` (caller keeps the heuristic result).
  - Gating: helper is only consulted when `PET_AGENT_LLM_GROUNDING=on`.
- [ ] **Step 2 (GREEN):** `llm_grounding.py`:
  - Serialize the scene graph (objects: id, class, attributes, position; relations) + the
    candidate ids + the utterance into a compact JSON prompt; request a structured
    `{object_id, justification}` via Ollama `format`. Validate `object_id ∈ candidates`.
    Lazy import, all-errors-→-None.
- [ ] **Step 3 (wire):** In `/command`, when `result.status ∈ {clarification, no_match}`
  **and** `PET_AGENT_LLM_GROUNDING=on`: call `llm_pick_target`. If it returns a valid id,
  build the `NavigationGoal` for that object (reuse the resolver's pose/explanation path)
  with `explanation` = the LLM justification, and proceed as success. Else keep the
  existing clarification/no_match behaviour. (LLM grounding is consulted *before* the
  clarification dialogue so a confident LLM pick short-circuits the question.)
- [ ] **Step 4:** `ruff`, run new tests + grounding/websocket suites, commit
  `feat(llm): LLM-assisted grounding writes justification into NavigationGoal.explanation`.

---

## Task 4 — Acceptance (≥5 scripted dialogues) + spec §14.6.4 status

**Files:** `tests/test_llm_dialogue_acceptance.py`, `docs/spec.md` §14.6.4.

- [ ] **Step 1:** `tests/test_llm_dialogue_acceptance.py` — ≥5 scripted end-to-end
  dialogues against a **deterministic fake Ollama client** (hermetic), exercising:
  (a) ambiguous "go to the box" with two boxes → discriminating question;
  (b) follow-up "the left one" → resolves to the correct box;
  (c) LLM-assisted grounding picks a target the heuristic left ambiguous;
  (d) unparseable free-form phrasing the rule parser misses → Ollama parses it;
  (e) Ollama unreachable → rule-parser fallback still answers.
  Run via the FastAPI `TestClient` so `/command` + `session_id` are covered.
- [ ] **Step 2 (live, opt-in):** Same five dialogues behind `@pytest.mark.live` against the
  real `qwen2.5-coder:7b`; report pass count honestly in the spec status (a 7B local model
  may miss edge phrasings — report the real number, don't massage it).
- [ ] **Step 3:** Append **Status — implemented** to spec §14.6.4: modules
  (`OllamaCommandParser` + `make_llm_parser`, `dialogue.py`, `llm_grounding.py`), selectors
  (`PET_AGENT_LLM_PARSER=on` + `PET_AGENT_LLM_BACKEND=ollama`, `PET_AGENT_LLM_GROUNDING=on`,
  `.[llm]` extra, `PET_AGENT_OLLAMA_MODEL/HOST`), the hermetic acceptance result, and the
  honest live pass-rate. Note every path has a deterministic fallback.
- [ ] **Step 4 (final verify):** `.venv/bin/ruff check . && .venv/bin/ruff format --check .
  && .venv/bin/pytest -q` — ruff clean, full suite green.
- [ ] **Step 5:** Commit `docs(llm): §14.6.4 local Ollama + clarification + grounding implemented`.

---

## Self-review checklist

- **Spec coverage (§14.6.4):** Ollama backend behind `PET_AGENT_LLM_BACKEND` with JSON-schema
  `format` + Pydantic gate (Task 1) ✓; multi-turn clarification with per-session state +
  discriminating questions (Task 2) ✓; LLM-assisted grounding → `NavigationGoal.explanation`
  gated by `PET_AGENT_LLM_GROUNDING` (Task 3) ✓; ≥5 scripted dialogues (Task 4) ✓.
- **Fallbacks:** missing `ollama` pkg, unreachable host, schema failure, timeout → rule
  parser / templated question / heuristic resolver. The default pytest suite is hermetic
  (no live model); live tests are `@pytest.mark.live`, opt-in.
- **Event-driven:** all LLM calls are on `/command` only — never in the perception/control
  loops.
- **No schema invention:** reuses `CommandIntent`, `GroundingResult`, `NavigationGoal`.
- **Honesty:** the live pass-rate of a local 7B model is reported as measured.

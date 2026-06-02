# 3D Pet Agent

`3d-pet-agent` 是一個實驗性的 3D 電腦寵物系統。一隻虛擬 3D 貓透過攝影機理解真實桌面 / 房間中的物體，並回應像「躲到鍵盤後面」或「走到紅色杯子的右邊但避開滑鼠」這類自然語言指令。

> **狀態：Phases 1–10 完成並驗證。** 242 個測試全綠；bundled evaluation dataset 達 100% task success rate。
>
> - 系統架構、模組職責、設計規則、各 phase 內部運作 → [`docs/architecture.md`](docs/architecture.md)
> - Phase 10 evaluation 結果 → [`docs/eval.md`](docs/eval.md)
> - 完整 v2 規格 → [`docs/spec.md`](docs/spec.md)

---

## 🚀 Getting Started

> Linux + RTX 4070 + Python 3.12 + Node 18。

```bash
git clone https://github.com/SoWiEee/3D-pet-agent.git
cd 3D-pet-agent

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# frontend
cd frontend && npm install && cd ..

# verify
.venv/bin/python -c "import torch; print('cuda:', torch.cuda.is_available())"
.venv/bin/pytest -q   # 預期 242 passed
```

> 首次跑 perception 時會自動從 Hugging Face 下載權重（GroundingDINO ≈ 700 MB、SAM ≈ 400 MB、Depth Anything V2 ≈ 100 MB）。

---

## 🚀 Quick Start (CLI)

所有 CLI 模式由 `main.py --mode <name>` 派發。各模式的輸出檔意義與內部運作見 [`docs/architecture.md`](docs/architecture.md)。

### Sandbox — 純寵物 runtime（不需感知模型）

```bash
# 單一目標移動
.venv/bin/python main.py --mode sandbox --target 0.5 0.0 1.2

# 腳本化動作（idle / move_to / look_at / set_emotion / play_animation / ask）
.venv/bin/python main.py --mode sandbox --script samples/pet_actions.jsonl
```

### Snapshot — 感知（單張圖片）

```bash
# 偵測 + 分割
.venv/bin/python main.py --mode snapshot \
  --image samples/desk.jpg \
  --prompts configs/prompts.txt \
  --out runs

# + 3D lifting（Depth Anything V2）
.venv/bin/python main.py --mode snapshot \
  --image samples/desk.jpg --lift --fov 60 --out runs

# + tracker + 持久 SemanticMap
.venv/bin/python main.py --mode snapshot \
  --image samples/desk.jpg --lift --track --frames 5 --out runs
```

> 自訂 prompts：`printf 'cat\nremote\ncouch\n' > /tmp/my.txt`，加 `--prompts /tmp/my.txt`。

### Eval — Phase 10 evaluation harness

```bash
.venv/bin/python main.py --mode eval \
  --dataset samples/eval_dataset.jsonl \
  --out runs
```

產出 `runs/eval_<timestamp>/{report.md, records.csv, records.jsonl}`，並在 task success rate < 50% 時以 non-zero 退出（適合接 CI）。預期結果見 [`docs/eval.md`](docs/eval.md)。

---

## End-to-End demo（後端 + 前端）

**Terminal 1 — 後端：**

```bash
.venv/bin/uvicorn src.runtime.websocket_server:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — 前端：**

```bash
cd frontend && npm run dev
```

瀏覽器開 `http://127.0.0.1:5173/`，右上角顯示 `● online`。

### 透過前端命令列

底部 input 按 Enter 送出：

- **自然語言指令**（Phase 6 parser → grounding → A\* → controller）：
  - `go to the cup`
  - `hide behind the keyboard but avoid the mouse`
  - `look at the monitor`
  - `explore the desk`
- **直接動作**：
  - `move 0.5 0 1.2`、`path 0 0 0 ; 0.3 0 0.5 ; 0.6 0 1.0`、`look -0.3 0.4 1.0`
  - `anim sit` / `emote curious` / `say hello`

---

## HTTP / WS 端點速查表

| Endpoint | 用途 |
|---|---|
| `GET  /healthz` | 健康檢查 |
| `GET  /pet/state` | 當前 PetState |
| `POST /pet/action` | 送一個 `PetAction`（`move_to` / `move_follow_path` / `look_at` / `play_animation` / `set_emotion` / `ask`）|
| `POST /pet/perception` | 餵 Phase 2 感知結果（placeholder 行為）|
| `POST /perception/lifted` | 餵 Phase 3/4 lifted 結果，廣播 `world_update` |
| `GET  /semantic/map` / `POST /semantic/reset` | 讀 / 清 SemanticMap |
| `GET  /scene/graph` | 當前場景圖 |
| `GET  /planning/occupancy` | Phase 7 occupancy grid debug |
| `POST /command` | **核心入口**：自然語言 → parse → ground → plan → control → `move_follow_path` |
| `GET  /control/last_trace` / `POST /control/simulate` | Phase 8 controller debug |
| `POST /exploration/observe` / `POST /exploration/step` / `GET /exploration/coverage` / `POST /exploration/reset` | Phase 9 探索 |
| `WS   /ws/pet` | 雙向動作串流（新連線會收到 PetState + 最近一次 `world_update`）|

---

## 設定檔

集中在 `configs/`，由 `src/config.py` 透過 pydantic 驗證：

| 檔案 | 用途 |
|---|---|
| `configs/models.yaml` | 模型 ID、device、box/text thresholds |
| `configs/thresholds.yaml` | grounding / tracking / relations / behavior |
| `configs/runtime.yaml` | 更新率、server host/port |
| `configs/navigation.yaml` | Phase 7：grid + planner + constraint halo |
| `configs/control.yaml` | Phase 8：kinematic + pure-pursuit + PID + preempt |
| `configs/prompts.txt` | 預設物件提示詞 |

環境變數覆寫使用 `PET_AGENT_` 前綴：

```bash
export PET_AGENT_DEVICE=cuda
export PET_AGENT_CAMERA_INDEX=0
```

---

## 開發工作流

```bash
.venv/bin/ruff check .                   # lint
.venv/bin/ruff format .                  # format
.venv/bin/pytest -q                      # 242 tests
.venv/bin/pytest tests/test_evaluation.py::test_runner_on_bundled_dataset_meets_threshold -v
cd frontend && npx vue-tsc --noEmit      # 前端型別檢查
```

---

## 已知限制

- 單目深度為相對深度，沒有相機標定時不具公制尺度。
- 透明 / 反射物體（杯子、瓶子）容易產生不穩定遮罩與深度。
- 開放詞彙偵測器對罕見或模糊指代可能失敗。
- 即時效能取決於 GPU VRAM、模型大小、影像解析度與更新率。
- OpenScene 風格 3D 語意查詢列為可選後端，不在第一輪 live demo 路徑。
- **GroundingDINO 以 fp32 執行**：fp16 在 deformable attention 的 `grid_sample` 觸發 dtype 不一致；RTX 4070 上 fp32 約 0.5–1 秒/張，足夠撐住 spec 的 2 Hz。


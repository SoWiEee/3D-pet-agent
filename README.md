# 3D Pet Agent

`3d-pet-agent` 是一個實驗性的 3D 電腦寵物系統。目標是讓一隻虛擬 3D 貓理解真實桌面或房間場景中的物體、推理它們的 3D 位置，並回應像「躲到鍵盤後面」或「走到紅色杯子的右邊」這類自然語言指令。

本專案整合了開放詞彙視覺模型、實例分割、深度估計、3D 場景圖，以及瀏覽器中的 3D 寵物 runtime。

> 狀態：Phase 1（3D 寵物 runtime + sandbox，含 path-following）、Phase 2（偵測 + 分割）、Phase 3（FramePacket + Depth Anything V2 + 3D 物件 lifting）已實作並驗證。後續導航 / 控制 / 探索路線圖請參考 [docs/spec.md](docs/spec.md)（v2，10 phases mainline + optional 擴充）。

---

## 目錄

- [動機](#動機)
- [系統架構](#系統架構)
- [Getting Started](#getting-started)
  - [系統需求](#系統需求)
  - [安裝步驟](#安裝步驟)
  - [Phase 1：3D 寵物 sandbox](#phase-13d-寵物-sandbox)
  - [Phase 2：感知主線（偵測 + 分割）](#phase-2感知主線偵測--分割)
  - [Phase 3：3D 物件 lifting（深度估計 + FramePacket）](#phase-33d-物件-lifting深度估計--framepacket)
  - [Phase 4：物件追蹤 + 持久 SemanticMap](#phase-4物件追蹤--持久-semanticmap)
  - [Phase 5–6：場景圖 + 命令 grounding](#phase-56場景圖--命令-grounding)
  - [Phase 7–8：A\* 規劃 + pure-pursuit 控制](#phase-78a-規劃--pure-pursuit-控制)
  - [Phase 9：主動探索](#phase-9主動探索)
  - [End-to-End 互動 demo（後端 + 前端）](#end-to-end-互動-demo後端--前端)
- [專案結構](#專案結構)
- [技術棧](#技術棧)
- [設定檔](#設定檔)
- [開發工作流](#開發工作流)
- [實作進度](#實作進度)
- [已知限制](#已知限制)
- [License](#license)
- [致謝](#致謝)

---

## 動機

多數虛擬寵物只活在螢幕裡。本專案探索一個更「接地」的版本：寵物可以透過攝影機觀察使用者的真實環境、理解可見物體、並依照物體做出反應。

範例互動：

```text
走到杯子那邊。
躲到鍵盤後面。
不要靠近水瓶。
看著我剛剛放在桌上的物體。
找一個在滑鼠和盒子之間的安全位置。
你正在看哪個物體？
```

核心想法：

```text
攝影機輸入 + 語言指令
  → 開放詞彙物件偵測
  → 物件遮罩
  → 深度估計
  → 3D 物件狀態
  → 場景圖
  → 指令 grounding
  → 寵物行為
```

---

## 系統架構

雙後端設計：

```
攝影機/影片 ─► 主線後端 (GroundingDINO → SAM → Depth → 3D lift)
                                                          │
RGB-D / 點雲 ─► 研究後端 (OpenScene-style 3D 查詢) ────────┤
                                                          ▼
                                                    3D 場景圖
                                                          │
使用者指令 ─► Command Parser ─► Grounding Resolver ◄──────┘
                                       │
                                       ▼
                               Behavior Planner ─► 3D 貓 Runtime（瀏覽器）
```

詳見 `CLAUDE.md` 與 `docs/spec.md §3`。

---

## Getting Started

以下步驟在 Linux + RTX 4070 + Python 3.12 + Node 18 環境下完整驗證過。

### 系統需求

| 項目 | 需求 |
|---|---|
| OS | Linux（主要目標；Windows 透過 WSL2 為次要選項） |
| GPU | NVIDIA GPU，建議 12 GB+ VRAM（RTX 4070 級或以上） |
| Python | 3.10 – 3.12 |
| Node.js | ≥ 18 |
| 套件管理 | [`uv`](https://docs.astral.sh/uv/) |
| 磁碟空間 | 模型權重首次下載約 2–3 GB |

### 安裝步驟

1. 取得專案並建立 Python 虛擬環境

```bash
git clone https://github.com/SoWiEee/3D-pet-agent.git
cd 3D-pet-agent
uv venv --python 3.12
source .venv/bin/activate
```

2. 安裝 Python 套件（含 torch、transformers、fastapi 等）

```bash
uv pip install -e ".[dev]"
```

3. 安裝前端依賴

```bash
cd frontend
npm install
cd ..
```

4.（選用）驗證環境是否就緒

```bash
.venv/bin/python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
.venv/bin/pytest tests/ -q
```

預期看到 `cuda: True` 與 `227 passed`。

---

### Phase 1：3D 寵物 sandbox

純寵物 runtime，不需要任何感知模型，可離線跑。對應 spec §4。

**單一目標移動：**

```bash
.venv/bin/python main.py --mode sandbox --target 0.5 0.0 1.2
```

預期輸出（最後一行為最終寵物狀態）：

```
mode=sandbox device=cuda
moving pet to (0.500, 0.000, 1.200)
final state: {"position":{"x":0.5,"y":0.0,"z":1.2},"animation":"walk", ...}
```

**腳本化動作（jsonl 連續播放 idle / move_to / look_at / set_emotion / play_animation / ask）：**

```bash
.venv/bin/python main.py --mode sandbox --script samples/pet_actions.jsonl
```

> `samples/pet_actions.jsonl` 內含 7 個動作；每個動作之間預設間隔 0.5 秒。

---

### Phase 2：感知主線（偵測 + 分割）

對單張圖片跑 GroundingDINO + SAM。**首次執行會自動從 Hugging Face 下載權重**（GroundingDINO ≈ 700 MB、SAM ≈ 400 MB），請保持網路連線。

```bash
.venv/bin/python main.py --mode snapshot \
  --image samples/desk.jpg \
  --prompts configs/prompts.txt \
  --out runs
```

完成後會在 `runs/` 下產生：

| 輸出 | 內容 |
|---|---|
| `runs/snapshot_<image>.json` | `PerceptionResult`：每個物件的 bbox、mask 路徑、信心分數、normalized center（符合 spec §5.4） |
| `runs/snapshot_<image>.png` | 視覺化結果：bbox + 標籤 + mask overlay |
| `runs/frame_000000/obj_XXX_<label>.png` | 每個物件的二值化遮罩 |

**自訂提示詞：** 換一張不是桌面場景的圖時，可以建立自己的 prompts：

```bash
printf 'cat\nremote control\ncouch\npillow\n' > /tmp/my_prompts.txt
.venv/bin/python main.py --mode snapshot \
  --image my_image.jpg --prompts /tmp/my_prompts.txt
```

> 預設的 `configs/prompts.txt` 是針對「桌面 / 房間」場景挑選的詞彙（cup / keyboard / mouse / monitor / …）。

---

### Phase 3：3D 物件 lifting（深度估計 + FramePacket）

在 snapshot mode 加 `--lift` 旗標即會啟用 Depth Anything V2 + 2D→3D lifting：

```bash
.venv/bin/python main.py --mode snapshot \
  --image samples/desk.jpg \
  --prompts /tmp/cat_prompts.txt \
  --out runs \
  --lift \
  --fov 60     # 估計相機水平 FOV（°），預設 60
```

完成後在 `runs/` 額外產生：

| 輸出 | 內容 |
|---|---|
| `runs/lifted_<image>.json` | 每個物件的 `ObjectState3D`：`center_3d_world`、`extent_3d`、`median_depth`、`depth_uncertainty`、`confidence`（含 detector / mask_quality / depth_quality / overall）|
| `runs/depth_<image>.png` | Inferno colormap 深度視覺化 |

**推到前端看 3D 中心點：** 啟動後端 + 前端，然後：

```bash
curl -X POST http://127.0.0.1:8000/perception/lifted \
  -H 'Content-Type: application/json' \
  -d @runs/lifted_desk.json
```

瀏覽器中會看到磷光綠的中心點 marker、垂直 stalk 到地面的光暈，以及右下角「WORLD OBJECTS」面板列出所有 lifted 物件與座標。

**注意**：

- 單目深度為**相對深度**（非公制），數值僅在同一張影像內可比。
- 沒提供相機標定時，從 `--fov` 估計 intrinsics；座標精度約為「同尺度的對」而非「公制」。
- 預設 `pose_source: fixed`（相機在 world 原點）；之後 Phase optional §14.1 可換成 ORB-SLAM。
- 深度模型載入時如 CUDA 不可用會自動 fall back 到 CPU（約 1–2 秒/張）。

---

### Phase 4：物件追蹤 + 持久 SemanticMap

在 `snapshot` 模式加 `--track`，會串接 Phase 3 lifter → IoU + class + 3D-distance 貪婪關聯 tracker（穩定 `track_NNN` id）→ 持久 SemanticMap（EMA 位置融合、Bayes 信心更新、`tracked → occluded → stale → lost` 狀態機）：

```bash
.venv/bin/python main.py --mode snapshot \
  --image samples/desk.jpg \
  --prompts configs/prompts.txt \
  --lift --track --frames 5 \
  --out runs
```

額外輸出：

| 輸出 | 內容 |
|---|---|
| `runs/semantic_map_<image>.json` | 持久 SemanticMap：每個 track 的最新 `ObjectState3D`、`confidence.overall`、`tracking_status`、`last_seen_frame` |

> 持久 map 是 byte-identical save→load，可隨時 `SemanticMap.load(path)` 接續上一輪追蹤。

---

### Phase 5–6：場景圖 + 命令 grounding

後端啟動後（見下方 demo 章節），SemanticMap 上每一次更新都會即時重算場景圖（11 種關係 — `left_of/right_of/in_front_of/behind/above/below/near/far_from/between/on_surface/occluding`，全部以平滑 ramp 評分），透過 `world_update` 廣播給前端的 RELATIONS 面板。

```bash
# 取得目前場景圖（pair + triple 關係，按分數排序）
curl http://127.0.0.1:8000/scene/graph | jq

# 送一句指令給後端
curl -X POST http://127.0.0.1:8000/command \
  -H 'Content-Type: application/json' \
  -d '{"text":"go to the cup"}'

# 帶關係的指令
curl -X POST http://127.0.0.1:8000/command \
  -d '{"text":"hide behind the keyboard but avoid the mouse"}'
```

`POST /command` 會完整跑：rule-based parser（10 種 `intent_type`）→ grounding resolver（`0.35·semantic + 0.20·attribute + 0.25·relation + 0.10·visibility + 0.10·feasibility`）→ 多候選時觸發 clarification ask、低信心時帶 explanation。

---

### Phase 7–8：A\* 規劃 + pure-pursuit 控制

成功 grounding 後，server 會：
1. 把 SemanticMap rasterise 成 XZ 平面 `OccupancyGrid`（包含 `obstacle_padding` 膨脹 + per-target 排除 + `avoid_object` halo）
2. 跑 8-connectivity A\*（Euclidean heuristic、no corner-cut、Bresenham LOS pruning）
3. 把 LOS-pruned 路徑送進 pure-pursuit 離線模擬器：`UnicycleState (x, y, θ)` + `v = clamp(base·cos²(he), v_min, v_max)` + `ω = Kp·he` + anti-windup PID 速度平滑 + slow-down radius
4. 廣播 `move_follow_path`，帶 dense 動力學可行軌跡 + `controller_trace`（`steps / duration_s / max_cross_track_error / max_heading_error / mean_speed`）給前端

```bash
# Debug：取得目前 occupancy grid（含 obstacle 膨脹後 blocked cells 數）
curl http://127.0.0.1:8000/planning/occupancy | jq '.blocked_cells'

# Debug：取得最近一次控制器 trace summary
curl http://127.0.0.1:8000/control/last_trace | jq

# 對任意路徑做離線控制模擬（不更動寵物狀態）
curl -X POST http://127.0.0.1:8000/control/simulate \
  -d '{"path":[[0,0,0],[1,0,1]],"start":[0,0,0],"start_theta":0.0}'
```

控制配置位於 `configs/control.yaml`（kinematic 限制、lookahead、PID 增益、preempt latency）。

---

### Phase 9：主動探索

`CoverageGrid` 用 uint16 計數器追蹤已觀察 / 未觀察 cells（與 nav grid 同框架），透過 vectorised cone sweep 更新；`ExplorationPlanner` 用 spec §12.1 啟發式（`0.40·new_area + 0.25·uncertainty + 0.20·search_relevance − 0.15·travel_cost`）在 4 種 goal 種類間挑下一個 viewpoint。

```bash
# 標記一個觀察 cone（攝影機在 (0,0)，朝 +x，FOV 90°，range 1 m）
curl -X POST http://127.0.0.1:8000/exploration/observe \
  -d '{"camera_xz":[0,0],"heading_rad":0,"fov_rad":1.5708,"range_m":1.0}'

# 取下一個探索 goal 並讓寵物走過去（會跑 planner + controller pipeline）
curl -X POST http://127.0.0.1:8000/exploration/step -d '{}'

# 自然語言指令同樣會 route 進來
curl -X POST http://127.0.0.1:8000/command \
  -d '{"text":"explore the desk and tell me what you found"}'

# 取得目前 coverage grid 與未觀察比率
curl http://127.0.0.1:8000/exploration/coverage | jq '.unobserved_ratio'
```

新發現的物件 id 會透過 `runtime.ask()` 在前端的對話泡泡裡報告。

---

### End-to-End 互動 demo（後端 + 前端）

啟動 FastAPI 後端與 Vite 前端，在瀏覽器中看到 3D 貓即時跟隨指令動作。

**Terminal 1 — 啟動後端：**

```bash
.venv/bin/uvicorn src.runtime.websocket_server:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — 啟動前端：**

```bash
cd frontend
npm run dev
```

接著用瀏覽器開啟：

```
http://127.0.0.1:5173/
```

連線後狀態列右上會顯示 `● online`，3D 視窗中央會出現一隻陶瓷色的貓。

**驅動寵物的幾種方式：**

- **底部命令列**輸入文字（按 Enter 送出）：
  - **自然語言指令**（會走 Phase 6 parser → grounding → A\* → controller）：
    - `go to the cup`
    - `hide behind the keyboard but avoid the mouse`
    - `look at the monitor`
    - `explore the desk`
  - **低階直接命令**：
    - `move 0.5 0 1.2` — 直接 tween 到指定座標
    - `path 0 0 0 ; 0.3 0 0.5 ; 0.6 0 1.0` — 沿給定路徑（chained Tween + smooth heading）
    - `look -0.3 0.4 1.0` — 看向某點
    - `anim sit` / `anim hide` / `anim curious` — 切換動畫
    - `emote curious` / `emote happy` — 切換情緒
    - `say hello` — 寵物說話

- **快捷按鈕**：「P1 · cup / P2 · keyboard / path · A\* / sit / hide / curious」。

- **HTTP API**（適合腳本化或測試）：
  ```bash
  curl -X POST http://127.0.0.1:8000/pet/action \
    -H 'Content-Type: application/json' \
    -d '{"action":"move_to","target_position_3d":[0.6,0,0.8]}'
  ```

- **將 Phase 2 感知結果送進寵物**（placeholder 行為：選最高信心物件並走過去）：
  ```bash
  curl -X POST http://127.0.0.1:8000/pet/perception \
    -H 'Content-Type: application/json' \
    -d @runs/snapshot_desk.json
  ```

**主要 HTTP / WS 端點：**

| Endpoint | 用途 |
|---|---|
| `GET  /healthz` | 健康檢查 |
| `GET  /pet/state` | 取得目前 PetState |
| `POST /pet/action` | 送一個 `PetAction`（`move_to` / `move_follow_path` / `look_at` / `play_animation` / `set_emotion` / `ask`）|
| `POST /pet/perception` | 餵 Phase 2 感知結果（placeholder 行為）|
| `POST /perception/lifted` | 餵 Phase 3/4 lifted 結果；server 端跑 tracker → SemanticMap → SceneGraph，廣播 `world_update` |
| `GET  /semantic/map` | 取得目前持久 SemanticMap |
| `POST /semantic/reset` | 清空 tracker + SemanticMap |
| `GET  /scene/graph` | 取得目前場景圖（pair + triple 關係，按分數排序）|
| `GET  /planning/occupancy` | Debug：取得 Phase 7 occupancy grid 快照 |
| `POST /command` | **核心入口**：自然語言指令 → parser → grounding → A\* → controller → `move_follow_path` |
| `GET  /control/last_trace` | 取得最近一次 controller trace summary |
| `POST /control/simulate` | 對任意路徑做離線 pure-pursuit 模擬（不更動寵物狀態）|
| `POST /exploration/observe` | 標記一個觀察 cone 進 CoverageGrid |
| `POST /exploration/step` | 跑一輪探索（pick goal → plan → controller → broadcast）|
| `GET  /exploration/coverage` | Debug：取得 CoverageGrid 快照 |
| `POST /exploration/reset` | 清空 CoverageGrid |
| `WS   /ws/pet` | 雙向動作串流（新連線會收到當前 PetState + 最近一次 `world_update`）|

---

## 專案結構

Phases 1–9 已實作的後端模組：

```text
3D-pet-agent/
├── main.py                      # CLI 入口
├── pyproject.toml               # uv-managed 套件 + ruff / pytest 設定
├── configs/
│   ├── models.yaml              # 模型 ID、device、閾值
│   ├── thresholds.yaml          # grounding / tracking / relations / behavior
│   ├── runtime.yaml             # 更新率、server host/port
│   ├── navigation.yaml          # Phase 7：grid、planner、constraints
│   ├── control.yaml             # Phase 8：kinematic、pure_pursuit、speed_pid
│   └── prompts.txt
├── src/
│   ├── config.py                # AppConfig（pydantic-settings，PET_AGENT_ 前綴）
│   ├── cli.py                   # --mode 派發；snapshot --lift --track
│   ├── camera_service/          # image_reader / video_reader / webcam
│   ├── perception/
│   │   ├── detector.py          # GroundingDINO 包裝
│   │   ├── segmenter.py         # SAM 包裝
│   │   ├── depth.py             # Depth Anything V2（lazy load + CPU fallback）
│   │   ├── pipeline.py          # run_frame_3d / run_frame_tracked
│   │   └── schema.py
│   ├── spatial/
│   │   ├── frame_packet.py      # FramePacket、CameraIntrinsics、CameraPoseWorld
│   │   ├── pose_source.py       # Fixed / Sim / SLAM pose sources
│   │   ├── object_lifter.py     # 2D mask → 3D centroid（percentile depth + 軸翻轉）
│   │   ├── semantic_map.py      # 持久 SemanticMap（EMA + Bayes + status machine）
│   │   ├── relation_scorer.py   # 11 種關係的 smooth-ramp 評分
│   │   └── scene_graph.py       # SceneGraphBuilder（pair + triple 走訪）
│   ├── tracking/
│   │   └── tracker.py           # IoU + class + 3D-distance 貪婪關聯
│   ├── language/
│   │   ├── schema.py            # CommandIntent / TargetSpec / RelationSpec
│   │   └── command_parser.py    # 10 種 intent 的 rule-based parser
│   ├── planning/
│   │   ├── schema.py            # NavigationGoal / NavigationConstraint
│   │   ├── grounding_resolver.py# Phase 6：候選評分 + clarification
│   │   ├── occupancy_grid.py    # XZ rasterise + obstacle 膨脹 + halo
│   │   ├── astar.py             # 8-conn A* + LOS pruning
│   │   └── planner.py           # Planner orchestrator
│   ├── control/
│   │   ├── kinematic.py         # frozen UnicycleState + kinematic_step
│   │   ├── pid.py               # immutable PID + anti-windup
│   │   ├── pure_pursuit.py      # lookahead + cos² 速度律
│   │   └── path_follower.py     # 離線模擬器 + ControlSummary
│   ├── exploration/
│   │   ├── coverage_grid.py     # uint16 觀察計數 + 未知區域 cluster
│   │   └── exploration_planner.py # 4 種 ExplorationGoal + §12.1 評分
│   └── runtime/
│       ├── pet_runtime.py       # PetState + 動作 API（含 move_follow_path、controller_trace）
│       └── websocket_server.py  # FastAPI app + 全部 endpoint
├── frontend/                    # Vue 3 + Vite + TypeScript + native Three.js
│   └── src/
│       ├── App.vue
│       ├── renderer/PetScene.ts # Three.js 場景，含 followPath chained tween
│       ├── composables/useWebSocket.ts
│       └── components/
│           ├── StatusBar.vue
│           ├── ModulePanel.vue
│           ├── Readouts.vue
│           ├── CommandBar.vue
│           ├── WorldObjectsLayer.vue
│           ├── RelationEdgesLayer.vue
│           ├── RegistrationMarks.vue
│           └── PetSpeech.vue
├── tests/                       # 227 個測試（unit + 整合 + server smoke）
│   ├── test_config.py
│   ├── test_pet_runtime.py
│   ├── test_cli.py
│   ├── test_perception_schema.py
│   ├── test_frame_packet.py
│   ├── test_object_lifter.py
│   ├── test_tracker.py
│   ├── test_semantic_map.py
│   ├── test_relation_scorer.py
│   ├── test_scene_graph.py
│   ├── test_command_parser.py
│   ├── test_grounding_resolver.py
│   ├── test_occupancy_grid.py
│   ├── test_astar.py
│   ├── test_planner.py
│   ├── test_kinematic.py
│   ├── test_pid.py
│   ├── test_pure_pursuit.py
│   ├── test_path_follower.py
│   ├── test_controller_server.py
│   ├── test_coverage_grid.py
│   ├── test_exploration_planner.py
│   └── test_exploration_server.py
├── .github/workflows/ci.yml     # backend + frontend CI（Phase 6）
├── samples/
│   ├── desk.jpg
│   └── pet_actions.jsonl
└── runs/                        # （gitignored）感知輸出、debug 影像
```

完整規劃結構見 `docs/spec.md §14`；模組層職責對應見 `CLAUDE.md`。

---

## 技術棧

### AI 與視覺

- PyTorch + CUDA（NVIDIA GPU）
- GroundingDINO（透過 `transformers` 的 `IDEA-Research/grounding-dino-tiny`）
- SAM（`facebook/sam-vit-base`）；Phase 3+ 可切換到 SAM 2
- Depth Anything V2（`depth-anything/Depth-Anything-V2-Small-hf`）
- OpenCV、Pillow、numpy、supervision

### 後端

- Python 3.12
- FastAPI + Uvicorn + websockets
- pydantic / pydantic-settings（型別化設定）
- `uv` 管理環境與依賴
- `ruff` lint + format
- `pytest` + `pytest-asyncio`

### 前端 / 3D Runtime

- Vue 3 + Vite + TypeScript
- 原生 Three.js（不使用 React wrapper）
- `@tweenjs/tween.js`（位移內插）
- `lil-gui`（debug 控制面板，預留）

---

## 設定檔

所有設定都集中在 `configs/`，由 `src/config.py` 透過 pydantic 驗證：

| 檔案 | 用途 |
|---|---|
| `configs/models.yaml` | 模型 ID、device、box/text thresholds |
| `configs/thresholds.yaml` | grounding / tracking / spatial relation / behavior 閾值 |
| `configs/runtime.yaml` | 感知 / tracking / renderer 更新率，server host/port |
| `configs/navigation.yaml` | Phase 7：grid 解析度與範圍、planner connectivity、constraint halo |
| `configs/control.yaml` | Phase 8：kinematic 限制、pure-pursuit 增益、PID、preempt latency |
| `configs/prompts.txt` | 預設物件提示詞清單 |

環境變數覆寫使用 `PET_AGENT_` 前綴：

```bash
export PET_AGENT_DEVICE=cuda
export PET_AGENT_CAMERA_INDEX=0
```

---

## 開發工作流

```bash
# Lint + format
.venv/bin/ruff check .
.venv/bin/ruff format .

# 單元測試
.venv/bin/pytest -q

# 跑單一測試
.venv/bin/pytest tests/test_pet_runtime.py::test_move_to_updates_state_and_broadcasts -v

# 前端型別檢查
cd frontend && npx vue-tsc --noEmit
```

建議的階段開發順序（對齊 docs/spec.md v2 §18）：

1. ✅ Phase 1 — 3D 寵物 sandbox（含 path-following）
2. ✅ Phase 2 — Snapshot 偵測 + 分割
3. ✅ Phase 3 — Depth + FramePacket + 3D 物件 lifting
4. ✅ Phase 4 — 物件追蹤 + SemanticMap（持久語意地圖）
5. ✅ Phase 5 — Scene Graph 與空間關係
6. ✅ Phase 6 — Command 解析與 Grounding Resolver → NavigationGoal
7. ✅ Phase 7 — Occupancy grid + A* path planning
8. ✅ Phase 8 — Pure-pursuit controller + 運動學模型
9. ✅ Phase 9 — Active exploration
10. ⬜ Phase 10 — Evaluation + demo packaging

Optional 擴充（不阻塞主線 demo）：

- Visual SLAM（ORB-SLAM2/3 替換 fixed pose）
- OpenScene 3D 開放詞彙查詢後端
- RL 探索策略
- ROS 2 Nav2 bridge

---

## 實作進度

| Phase | 描述 | 狀態 |
|---|---|---|
| 1 | 3D 寵物 runtime + sandbox（含 `move_follow_path`） | ✅ 完成 |
| 2 | 互動主線感知（GroundingDINO + SAM） | ✅ 完成 |
| 3 | Depth + FramePacket + 3D 物件 lifting | ✅ 完成（`snapshot --lift` + 前端 marker）|
| 4 | 物件追蹤 + SemanticMap（持久語意地圖） | ✅ 完成（`snapshot --track`，byte-identical save/load）|
| 5 | Scene Graph 與空間關係 | ✅ 完成（11 種關係 + RELATIONS 面板）|
| 6 | Command 解析 + Grounding Resolver → NavigationGoal | ✅ 完成（10 種 intent，rule-based parser + LLM seam）|
| 7 | Occupancy grid + A* path planning | ✅ 完成（8-conn、LOS smoothing、結構化失敗）|
| 8 | Pure-pursuit controller + 運動學模型 | ✅ 完成（unicycle + PID + 離線模擬 + controller_trace）|
| 9 | Active exploration | ✅ 完成（CoverageGrid + 4 種 ExplorationGoal）|
| 10 | Evaluation + demo packaging | ⬜ 待實作 |
| opt | Visual SLAM / OpenScene / RL / ROS 2 Nav2 | ⬜ optional 擴充 |

**測試覆蓋：** 227 個測試全綠（每個 phase 自帶 unit + 整合 + server smoke），ruff check + format 全綠，vue-tsc clean。

---

## 已知限制

- 單目深度為相對深度，沒有相機標定時不具公制尺度。
- 透明 / 反射物體（杯子、瓶子）易產生不穩定的遮罩與深度。
- 開放詞彙偵測器在罕見物體或模糊指代上可能失敗。
- 即時效能高度依賴 GPU VRAM、模型大小、影像解析度與更新率。
- OpenScene 風格 3D 語意查詢列為可選研究後端，不在第一輪 live demo 路徑中。
- **GroundingDINO 目前以 fp32 執行**：fp16 會在 deformable attention 的 `grid_sample` 觸發 dtype 不一致；在 RTX 4070 上 fp32 約 0.5–1 秒/張，已足以撐住 spec 設定的 2 Hz 感知頻率。

---

## License

License 細節尚未敲定。

---

## 致謝

本專案基於開源電腦視覺與 3D 圖學社群的成果，包括：GroundingDINO、Segment Anything、SAM 2、Depth Anything V2、OpenScene、PyTorch、OpenCV、Open3D、Vue、Vite、Three.js。

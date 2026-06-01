# 3D Pet Agent

`3d-pet-agent` 是一個實驗性的 3D 電腦寵物系統。目標是讓一隻虛擬 3D 貓理解真實桌面或房間場景中的物體、推理它們的 3D 位置，並回應像「躲到鍵盤後面」或「走到紅色杯子的右邊」這類自然語言指令。

本專案整合了開放詞彙視覺模型、實例分割、深度估計、3D 場景圖，以及瀏覽器中的 3D 寵物 runtime。

> 狀態：Phase 1（3D 寵物 runtime + sandbox）與 Phase 2（互動主線感知：偵測 + 分割）已實作並驗證。完整 10 階段藍圖請參考 [docs/spec.md](docs/spec.md)。

---

## 目錄

- [動機](#動機)
- [系統架構](#系統架構)
- [Getting Started](#getting-started)
  - [系統需求](#系統需求)
  - [安裝步驟](#安裝步驟)
  - [Phase 1：3D 寵物 sandbox](#phase-13d-寵物-sandbox)
  - [Phase 2：感知主線（偵測 + 分割）](#phase-2感知主線偵測--分割)
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
git clone https://github.com/<your-user-or-org>/3D-pet-agent.git
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

預期看到 `cuda: True` 與 `23 passed`。

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
  - `move 0.5 0 1.2` — 移動到指定座標
  - `look -0.3 0.4 1.0` — 看向某點
  - `anim sit` / `anim hide` / `anim curious` — 切換動畫
  - `emote curious` / `emote happy` — 切換情緒
  - `say hello` — 寵物說話（會以斜體 serif 浮現）

- **快捷按鈕**：「P1 · cup / P2 · keyboard / sit / hide / curious」。

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

> 完整 grounding（理解「躲到 X 後面」這類關係指令）屬於 Phase 6，目前尚未實作。

**主要 HTTP / WS 端點：**

| Endpoint | 用途 |
|---|---|
| `GET  /healthz` | 健康檢查 |
| `GET  /pet/state` | 取得目前 PetState |
| `POST /pet/action` | 送一個 `PetAction`（move_to / look_at / play_animation / set_emotion / ask）|
| `POST /pet/perception` | 餵 Phase 2 感知結果，placeholder 行為會驅動寵物 |
| `WS   /ws/pet` | 雙向動作串流（前端訂閱用）|

---

## 專案結構

目前已實作（Phase 1 + 2）的檔案：

```text
3D-pet-agent/
├── main.py                      # CLI 入口
├── pyproject.toml               # uv-managed 套件 + ruff / pytest 設定
├── configs/
│   ├── models.yaml              # 模型 ID、device、閾值（spec §16.1）
│   ├── thresholds.yaml          # grounding / tracking / relations / behavior
│   ├── runtime.yaml             # 更新率、server host/port
│   └── prompts.txt              # 預設桌面詞彙
├── src/
│   ├── config.py                # AppConfig（pydantic-settings，PET_AGENT_ 前綴）
│   ├── cli.py                   # --mode 派發
│   ├── camera_service/          # image_reader / video_reader / webcam
│   ├── perception/
│   │   ├── detector.py          # GroundingDINO 包裝
│   │   ├── segmenter.py         # SAM 包裝
│   │   ├── depth.py             # Depth Anything V2（Phase 3 接入用，已預備）
│   │   ├── pipeline.py          # 偵測 → 分割 → ObjectCandidate2D
│   │   └── schema.py            # PerceptionResult / ObjectCandidate2D（spec §5.4）
│   └── runtime/
│       ├── pet_runtime.py       # PetState + 動作 API（spec §4.3）
│       └── websocket_server.py  # FastAPI app + /ws/pet
├── frontend/                    # Vue 3 + Vite + TypeScript + native Three.js
│   └── src/
│       ├── App.vue
│       ├── renderer/PetScene.ts # Three.js 場景（貓、地板網格、目標標記、tween）
│       ├── composables/useWebSocket.ts
│       └── components/
│           ├── StatusBar.vue
│           ├── ModulePanel.vue
│           ├── Readouts.vue
│           ├── CommandBar.vue
│           ├── RegistrationMarks.vue
│           └── PetSpeech.vue
├── tests/                       # 23 個單元測試
│   ├── test_config.py
│   ├── test_pet_runtime.py
│   ├── test_cli.py
│   └── test_perception_schema.py
├── samples/
│   ├── desk.jpg                 # COCO 測試圖
│   └── pet_actions.jsonl        # sandbox 腳本範例
└── runs/                        # （gitignored）感知輸出、debug 影像
```

完整規劃結構（含 Phase 3+ 模組）見 `docs/spec.md §14`。

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

建議的階段開發順序（對齊 docs/spec.md §19）：

1. ✅ Phase 1 — 3D 寵物 sandbox
2. ✅ Phase 2 — Snapshot 偵測 + 分割
3. ⬜ Phase 3 — 深度估計與 3D 物件 lifting
4. ⬜ Phase 4 — 物件記憶與追蹤
5. ⬜ Phase 5 — 場景圖與空間關係
6. ⬜ Phase 6 — 指令解析與 Grounding Resolver
7. ⬜ Phase 7 — 行為規劃
8. ⬜ Phase 8 — Live demo 模式
9. ⬜ Phase 9 — 評估與報告
10. ⬜ Phase 10 — OpenScene 研究後端（可選）

---

## 實作進度

| Phase | 描述 | 狀態 |
|---|---|---|
| 1 | 3D 寵物 runtime + sandbox | ✅ 完成 |
| 2 | 互動主線感知（偵測 + 分割） | ✅ 完成 |
| 3 | 深度估計與 3D 物件 lifting | ⬜ 模組已預備、尚未啟用 |
| 4 | 物件追蹤與時序穩定 | ⬜ 待實作 |
| 5 | 3D 場景圖與空間關係 | ⬜ 待實作 |
| 6 | 指令解析與 Grounding Resolver | ⬜ 待實作（目前以最高信心物件作為 placeholder 行為）|
| 7 | 寵物行為規劃 | ⬜ 待實作 |
| 8 | OpenScene 研究後端 | ⬜ 待實作 |
| 9 | 雙後端比較 | ⬜ 待實作 |
| 10 | 評估、demo 協定、報告資產 | ⬜ 待實作 |

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

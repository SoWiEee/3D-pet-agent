# Project Review — 待改進清單

> 更新：2026-06-02。對齊 commit `7997d0d`（Phase 10 + UI 重構完成後）。
>
> 主線 Phase 1–10 規格已全部實作並驗證（242 tests 綠、bundled eval
> dataset 達 100% task success）。本文件記錄**規格存在但實作有缺口**、
> **品質可精進**、**UX 待補**、以及 spec §14 列為 optional 的擴充項目。

---

## A. 規格存在但實作有缺口

對應 `docs/spec.md` 已明文要求、但目前實作沒到位的部分。

### A1. Live demo loop（高 CP 值）

**現狀：** `main.py --mode demo` 只啟動 FastAPI server，沒有 webcam →
perception → tracker → SemanticMap → broadcast 的 2 Hz 背景迴圈。
目前只能透過 `curl -d @samples/desk_scene_demo.json
http://127.0.0.1:8000/perception/lifted` 灌入預備好的 lifted JSON。

**影響：** 拿真實攝影機跑 demo 時整條感知 → 控制 pipeline 無法 end-to-end。
這是現場展示最容易卡住的地方。

**修補方向：**
- 在 `src/runtime/websocket_server.py` 起一個 asyncio task
- 用 `camera_service.webcam` 抓 frame（沿用 `PerceptionPipeline.run_frame_tracked`）
- 每 ~500 ms 觸發一次 server 端的 tracker + SemanticMap.update + broadcast
- 用 `configs/runtime.yaml::runtime.perception_update_hz` 控制頻率
- 加入 `pet/perception/start` / `pet/perception/stop` 端點供前端控制

**成本：** 中（不影響既有 endpoint，但要小心 perception 模型載入 / VRAM 管理）

---

### A2. LLM parser seam（中 CP 值）

**現狀：** `PET_AGENT_LLM_PARSER=on` 環境變數已預留，但 `src/language/
command_parser.py::parse_command` 只走 `RuleCommandParser`。Rule parser
處理 10+ 種 canonical 句型；遇到自由表述（"可以走去左邊那個盒子那邊嗎"、
"sit next to the lamp"）會 fall back 為 `unparseable`。

**影響：** Demo 時觀眾常會說「自然」的句子，rule parser 命中率有限。

**修補方向：**
- 在 `command_parser.py` 加 `LLMCommandParser` class
- Tool use / structured outputs 強制回傳 `CommandIntent` JSON
- Schema 驗證失敗 → 自動 fall back 到 `RuleCommandParser`
- 加 timeout 限制（避免阻塞）
- 用 `PET_AGENT_LLM_PARSER=on` 開關

**成本：** 中（API key 管理 + 整合測試）

---

### A3. Phase 10 dataset 規模

**現狀：** `samples/eval_dataset.jsonl` 只有 8 trial。Spec §13.3 規定：

```
50 自然語言指令
10+ ambiguous / failure 指令
5   no-target 指令
```

合計 65 trial，目前不到 8/65。

**影響：** Eval 數字（100% task success）統計顯著性低，failure gallery 樣本少。

**修補方向：**
- 補齊 6 種 spec §13.1 demo scenarios 的變體
- 加入更多場景（5 room arrangements、10 desk scenes 各自的指令）
- 加入「同義詞」變體（go to / approach / walk to / move toward）
- 加入「multi-step」（先走過去再 sit）— 目前 parser 不支援，會成為 failure 案例

**成本：** 低（純資料工作）

---

### A4. Collision counting 精度

**現狀：** `src/evaluation/runner.py::_count_collisions` 只取 waypoint 取樣
然後查 occupancy cell，是否 blocked。

**問題：** 長 segment 中段穿過 obstacle 不會被計數（取樣點在兩端、obstacle 在中間）。

**修補方向：**
- 重用 `src/planning/astar.py::line_of_sight`（已有 Bresenham 實作）
- 對每兩個相鄰 waypoint 之間的 line 做 LOS 檢查
- 任何 line 經過的 blocked cell 都算一次 collision

**成本：** 低（重用既有函式）

---

### A5. Preempt latency 量測

**現狀：** Spec §11.4 規定新指令必須在 100 ms 內 preempt 舊路徑。
前端的 chained tween 確實會被下一個 `move_follow_path` 取代，但：
- 沒有任何 timer / metric 量測這個 latency
- 沒有 unit test 或 e2e test 驗證 < 100 ms
- 沒有 spec §3.8 EvaluationRecord 欄位記錄

**修補方向：**
- 在 `PetAction` 加 `preempt_latency_ms` 欄位（optional）
- 前端 receive 新 `move_follow_path` 時對比上次 broadcast 時間
- Eval harness 加 preempt 場景 + assertion

**成本：** 低（純儀器化 + 一條 assertion）

---

## B. 實作品質可精進

工作正常但可以做得更好。

### B1. Tracker 演算法

**現狀：** `src/tracking/tracker.py::_associate` 是貪婪的
IoU + class + 3D-distance 匹配。

**精進方向：** ByteTrack 的 cascade matching（高信心 → 低信心兩階段）
+ Kalman filter predict step。`_associate` 是單一方法，預留好替換槽。

**成本：** 中（演算法 + 整合測試）

---

### B2. SemanticMap 自動載入

**現狀：** `SemanticMap.save → load → save` byte-identical 已驗證，
但 server boot 時不會自動載入上一輪 `runs/semantic_map_*.json`。
每次 `uvicorn` 重啟，場景就清空。

**精進方向：**
- 在 `websocket_server.py` lifespan startup 檢查 `runs/last_map.json`
- 存在則 `SemanticMap.load()` 取代空 map
- 加 `POST /semantic/save` 端點手動觸發 snapshot

**成本：** 低

---

### B3. WebSocket auto-reconnect

**現狀：** 前端 `useWebSocket.ts` 連線中斷後不會自動重連。

**精進方向：** 指數退避重連 + 重連成功時自動 re-subscribe（server 端
`last_world_update` 會回放，前端不用重請求 SemanticMap）。

**成本：** 低

---

### B4. Cat 動畫深度

**現狀：** `PetScene.ts::Cat` 只有：
- Vertical bob（呼吸）
- Tail sway（idle/walk 不同幅度）
- Ears flicker

**精進方向：**
- 走路時的腿部 IK / 4 隻腳輪流
- `hide` 真的低身、`sit` 真的坐下
- 6 種 emotion 各自有 idle 動作 / 表情變化

**成本：** 高（需要 rig / morph target；或改用預先做好的 GLTF skeletal animation）

---

### B5. Speech bubble queue

**現狀：** `PetSpeech` 直接顯示最新 `petState.speech`。快速連發
`runtime.ask` 會互蓋，例如 grounding 的 explanation 講完前
clarification 又 push 進來。

**精進方向：**
- Server 端把 ask 排成 queue，依序廣播
- 或前端 buffer 顯示時間（每段最少 1.5 s）

**成本：** 低

---

### B6. CI 實際驗證

**現狀：** `.github/workflows/ci.yml` 寫好（backend pytest + ruff、
frontend vite build + vue-tsc），但只在 local 跑過 pytest。沒在實際
GitHub Actions 環境 push 過。

**精進方向：** Push 一次 PR / 觸發 workflow，看 cold cache 環境下
能否完成 `uv pip install -e ".[dev]"` 與 `npm ci`，調整 timeout。

**成本：** 低（觸發即知）

---

## C. UX / 視覺化精進

| 項目 | 現狀 | 修補方向 |
|---|---|---|
| **A\* 路徑視覺化** | `/planning/occupancy` 已回傳 grid，前端無 overlay | 在地板加一層 instanced mesh，blocked cells 顯紅色，path waypoints 連線顯磷光綠 |
| **CoverageGrid 視覺化** | Phase 9 `/exploration/coverage` 有資料，前端無熱圖 | 同上做地板 overlay，未觀察 cell 顯灰、已觀察按計數做漸層 |
| **Exploration goal marker** | 導航成功有 target marker；探索 goal 沒有視覺目標 | 重用 `TargetMarker` class，加 `kind: "exploration"` 樣式變體 |
| **Eval replay UI** | 跑 dataset 只能看 markdown report | 加前端 "/eval" 路由，可逐 trial step through、播 controller_trace 動畫 |
| **拖曳放物件** | 沒有 UI 編輯場景，只能 curl | 加「點地板放物件」+ class label 下拉，發 POST 到 `/perception/lifted` |
| **Grounding explanation panel** | `goal.explanation` 只透過 speech bubble 短暫顯示 | 在 topbar 加「上一次推理」按鈕 → 彈窗顯示 parser intent + resolver score breakdown + planner status |
| **Path failure overlay** | A\* 失敗只透過 speech 講；occupancy 為何 blocked 看不到 | 失敗時把目前的 `OccupancyGrid` overlay 自動打開 5 s |

---

## D. Optional sidecars (spec §14)

**規格明文標示 optional，不阻塞 demo。** 暫時不做也沒關係。

1. **Visual SLAM**（§14.1）— ORB-SLAM2/3 取代 `FixedPoseSource`，
   把 `pose_source: fixed` → `pose_source: slam`，camera 移動時
   world frame 仍穩定。
2. **OpenScene research backend**（§14.2）— 第二感知 backend，
   讓 `source_backend ∈ {mainline_grounding_sam, openscene}` tag
   真的有第二個選項；附 backend comparison report。
3. **RL exploration policy**（§14.3）— 取代 Phase 9 的啟發式評分。
4. **ROS 2 Nav2 bridge**（§14.4）— Bridge 到真實機器人。

---

## 建議優先順序

按 **CP 值 = (對 demo 影響) / (修補成本)** 排：

1. **A1 Live demo loop** — 不做就無法 end-to-end 跑真實攝影機
2. **A3 Dataset 擴充** — 純資料工作、立刻提升 eval 統計顯著性
3. **B2 SemanticMap 自動載入** — restart 場景不消失，demo 順序更自然
4. **A4 Collision counting** — 重用既有函式、改善 eval 精度
5. **C 系列任選 1–2 項視覺化** — 拉高 demo 觀感
6. **A2 LLM parser** — 提升自然語句命中率
7. **B 其他項** + D 系列依預算與目標決定

---

## 何時更新本文件

- 完成任一項時，移到 `## 已完成` 區段（或直接從本文件刪除）
- 發現新的 known issue 時補進對應分類
- 修補方向有更具體計畫時，把 bullet 換成 link 到 commit / PR

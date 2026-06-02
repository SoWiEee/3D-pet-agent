# Project Review — 待改進清單

> 更新：2026-06-02。對齊 commit `7997d0d`（Phase 10 + UI 重構完成後）。
>
> 主線 Phase 1–10 規格已全部實作並驗證（242 tests 綠、bundled eval
> dataset 達 100% task success）。本文件記錄**規格存在但實作有缺口**、
> **品質可精進**、**UX 待補**、以及 spec §14 列為 optional 的擴充項目。

---

## A. 規格存在但實作有缺口

對應 `docs/spec.md` 已明文要求、但目前實作沒到位的部分。

### ✅ A1. Live demo loop — 完成（commit `7bdfdb8`）

`src/runtime/perception_loop.py` 已實作；`POST /perception/start /stop /status`
端點上線；5 unit + 2 server smoke tests 全綠。Heavy 模型 lazy load，opt-in。

---

### A1. Live demo loop（高 CP 值，原始描述）

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

### ✅ A2. LLM parser seam — 完成（commit `b4e6b41`）

`src/language/llm_parser.py::LLMCommandParser` 已實作；Anthropic SDK lazy import，
client_factory 注入點讓測試免 API key。所有失敗路徑（import / network /
timeout / schema validation）→ silent fallback to rule parser。10 tests 全綠。

---

### A2. LLM parser seam（中 CP 值，原始描述）

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

### ✅ A3. Phase 10 dataset 規模 — 完成

`samples/eval_dataset.jsonl` 已從 8 trial 擴充到 **65 trial**，由
`scripts/build_eval_dataset.py` 程式化生成。最新分佈：navigate 25 / clarification 13 /
look_at 9 / no_match 7 / hide 6 / explore 2 / stop 2 / report 1。
完整滿足 spec §13.3（50+ NL、10+ ambiguous、5 no-target）。

最新 run：65 trials、task success rate **100%**、0 collisions、mean latency 4.3 ms。
詳見 [docs/eval.md](eval.md)。

---

### A3. Phase 10 dataset 規模（原始描述）

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

### ✅ B1. Tracker 演算法 — 完成（commit `52c68ce`）

`src/tracking/tracker.py` 改為 ByteTrack 兩階段 cascade：先用高信心偵測對所有
活躍 track 配對，剩下未配對的 track 再對低信心偵測補一輪。低信心 box 只能「續命」
既有 track，永不開新 track（faint box 多半是瞬間遮擋／模糊）。每個 track 帶 3D
中心與 2D bbox 的 EMA 速度，配對時比對「預測位姿」而非上一幀位姿，快速等速運動下
id 不跳。高/低分界用 `max(detector, overall)`，避免 `/perception/lifted`、eval
runner 只填 `overall` 的物件被丟掉。新增 3 tests（低信心不開 track／低信心續命／
速度模型保 id），既有 7 tests 全綠。

---

### B1. Tracker 演算法（原始描述）

**現狀：** `src/tracking/tracker.py::_associate` 是貪婪的
IoU + class + 3D-distance 匹配。

**精進方向：** ByteTrack 的 cascade matching（高信心 → 低信心兩階段）
+ Kalman filter predict step。`_associate` 是單一方法，預留好替換槽。

**成本：** 中（演算法 + 整合測試）

---

### ✅ B2. SemanticMap 自動載入 — 完成（commit `32c5ab0`）

`src/runtime/websocket_server.py::_try_autoload_semantic_map` 在 lifespan
startup 從 `PET_AGENT_SEMANTIC_MAP_PATH`（預設 `runs/last_map.json`）載入
快照、in-place 替換內容並廣播一次 `world_update`。`POST /semantic/save`
端點觸發儲存（預設路徑、可帶 `path` 覆寫）。Corrupt / 缺檔 / load 例外 → silent
fallback to clean boot。原 SemanticMap reference 不換，避免 perception_loop
等模組失效。7 tests 全綠。

---

### B2. SemanticMap 自動載入（原始描述）

**現狀：** `SemanticMap.save → load → save` byte-identical 已驗證，
但 server boot 時不會自動載入上一輪 `runs/semantic_map_*.json`。
每次 `uvicorn` 重啟，場景就清空。

**精進方向：**
- 在 `websocket_server.py` lifespan startup 檢查 `runs/last_map.json`
- 存在則 `SemanticMap.load()` 取代空 map
- 加 `POST /semantic/save` 端點手動觸發 snapshot

**成本：** 低

---

### ✅ B3. WebSocket auto-reconnect — 完成（commit `a6eb9cc`）

`useWebSocket.ts` 重連改為指數退避（base 500ms、cap 10s）＋ full jitter，多分頁
不會同步重試。`onclose` 不再於元件卸載後排重連：`onBeforeUnmount` 設 `disposed`
旗標、拆掉 `onclose`/`onerror` 後才 close。Server 端 `ws_pet` 在每次新連線都會回放
pet state ＋ 最新 `world_update`，所以重連本身就會還原場景，前端不需重新訂閱。
另外 export `reconnectAttempts` 供 UI 使用。

---

### B3. WebSocket auto-reconnect（原始描述）

**現狀：** 前端 `useWebSocket.ts` 連線中斷後不會自動重連。

**精進方向：** 指數退避重連 + 重連成功時自動 re-subscribe（server 端
`last_world_update` 會回放，前端不用重請求 SemanticMap）。

**成本：** 低

---

### ✅ B4. Cat 動畫深度 — 完成（commit `bd48482`）

`Cat` 抽成獨立模組 `frontend/src/renderer/Cat.ts`，純程序化（不需 rig）深化動畫。
姿態是一組由 animation ＋ emotion 解析出的 eased 標量：
- walk/run 用對角四腳步態（hip-pivot 擺動）＋ footfall 下沉
- `sit` 真的坐下（後低、前抬、後腿收）、`hide` 真的低身（趴平、耳朵後貼、尾巴收）
- happy/curious/confused/scared/playful 各自疊加耳朵／尾巴／歪頭／顫抖表情線索

所有標量以與幀率無關的方式 lerp，狀態間平滑過渡。已用 headless 截圖驗證場景無
runtime error 正常渲染。

---

### B4. Cat 動畫深度（原始描述）

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

### ✅ A\* 路徑 / CoverageGrid / Exploration goal marker — 完成（commit `68fde2a`）

三個 scene overlay，皆由既有 pipeline 資料驅動（`frontend/src/renderer/PetScene.ts`）：

- **A\* 路徑視覺化**：直接畫 controller 的 dense `move_follow_path` 軌跡，做成
  additive-blend 發光 tube（`LineBasicMaterial` 在 WebGL 被鎖 1px、太淡，故改用
  `TubeGeometry`）＋ 起點/終點 puck。`followPath` 時顯示、抵達時淡出。比 rasterize
  occupancy grid 更貼近實際走的路徑。
- **CoverageGrid 視覺化**：`GET /exploration/coverage` 用 `CanvasTexture` 畫成地板
  熱圖（未觀察＝暖色霧、已觀察＝依計數的磷光漸層），單一貼圖 plane。視窗右上「覆蓋圖」
  按鈕切換；coverage 不走 WS 串流，所以是 on-demand fetch（並加進 vite dev proxy）。
- **Exploration goal marker**：依 goal kind 著色的 beacon（ring＋beam＋kind/score
  標籤）。Goal 透過新的 `PetAction.exploration_goal` 欄位搭 `move_follow_path` 廣播
  （`controller_trace` 因 Pydantic `extra="ignore"` 被默默丟掉，故新欄位有正式宣告），
  穿過 `runtime.move_follow_path`。已用 live backend ＋ headless 瀏覽器端到端驗證。

### C 系列（原始描述）

| 項目 | 現狀 | 修補方向 |
|---|---|---|
| ✅ **A\* 路徑視覺化** | ~~`/planning/occupancy` 已回傳 grid，前端無 overlay~~ → 完成（發光 tube，畫 controller 實際軌跡） | 在地板加一層 instanced mesh，blocked cells 顯紅色，path waypoints 連線顯磷光綠 |
| ✅ **CoverageGrid 視覺化** | ~~Phase 9 `/exploration/coverage` 有資料，前端無熱圖~~ → 完成（CanvasTexture 熱圖＋切換鈕） | 同上做地板 overlay，未觀察 cell 顯灰、已觀察按計數做漸層 |
| ✅ **Exploration goal marker** | ~~導航成功有 target marker；探索 goal 沒有視覺目標~~ → 完成（kind 著色 beacon，goal 走 WS 廣播） | 重用 `TargetMarker` class，加 `kind: "exploration"` 樣式變體 |
| **Eval replay UI** | 跑 dataset 只能看 markdown report | 加前端 "/eval" 路由，可逐 trial step through、播 controller_trace 動畫 |
| **拖曳放物件** | 沒有 UI 編輯場景，只能 curl | 加「點地板放物件」+ class label 下拉，發 POST 到 `/perception/lifted` |
| ✅ **Grounding explanation panel** | ~~`goal.explanation` 只透過 speech bubble 短暫顯示~~ → 完成（commit `a010912`） | topbar「上一次推理」按鈕 → 彈窗顯示解析意圖＋每候選評分拆解堆疊條（resolver 現會輸出 `candidate_breakdowns`）＋planner status |
| ✅ **Path failure overlay** | ~~A\* 失敗只透過 speech 講；occupancy 為何 blocked 看不到~~ → 完成（commit `a010912`） | 規劃失敗（plan_failed / no_path / goal_unreachable / start_blocked）時，自動 fetch `/planning/occupancy` 並把 blocked cells 以紅色地板貼圖顯示 5 s |

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

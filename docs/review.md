# Project Review — 待改進清單

> 更新：2026-06-03。已完成項目已自本文件移除（紀錄在 git 歷史 / commit message）。
>
> 主線 Phase 1–10 規格已全部實作並驗證。本文件只列**尚未實作**的部分：
> 規格存在但有缺口、品質可精進、UX 待補，以及 spec §14 列為 optional 的擴充項目。

---

## A. 規格存在但實作有缺口

對應 `docs/spec.md` 已明文要求、但目前實作沒到位的部分。

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

### B6. CI 實際驗證

**現狀：** `.github/workflows/ci.yml` 寫好（backend pytest + ruff、
frontend vite build + vue-tsc），近期已透過 push 觸發並修過數次
ruff-format 失敗，但尚未確認 cold-cache 環境下完整安裝步驟的 timeout。

**精進方向：** 檢查 cold cache 環境下 `uv pip install -e ".[dev]"` 與
`npm ci` 是否能在 timeout 內完成，必要時調整。

**成本：** 低（觸發即知）

---

## C. UX / 視覺化精進

| 項目 | 現狀 | 修補方向 |
|---|---|---|
| **Eval replay UI** | 跑 dataset 只能看 markdown report | 加前端 "/eval" 路由，可逐 trial step through、播 controller_trace 動畫 |

---

## D. Optional sidecars (spec §14)

**規格明文標示 optional，不阻塞 demo。** 暫時不做也沒關係。

1. **OpenScene research backend**（§14.2）— 第二感知 backend，
   讓 `source_backend ∈ {mainline_grounding_sam, openscene}` tag
   真的有第二個選項；附 backend comparison report。
2. **RL exploration policy**（§14.3）— 取代 Phase 9 的啟發式評分。
3. **ROS 2 Nav2 bridge**（§14.4）— Bridge 到真實機器人。

> ✅ **Visual SLAM**（§14.1）已實作 — `src/research/slam_adapter.py`
> （ORB 視覺里程計，RGB-D PnP / 單目 essential，輸出 graphics-world
> `world ← camera` pose）。`PET_AGENT_POSE_SOURCE=slam` 在 perception loop
> 啟用。注意：frame-to-frame VO，無 loop closure / global BA，長迴圈會漂移；
> 真正的 ORB-SLAM3 / DROID-SLAM 可沿 `VisualOdometry` protocol 替換。

---

## 建議優先順序

按 **CP 值 = (對 demo 影響) / (修補成本)** 排：

1. **A5 Preempt latency** — 小儀器化 + 一條 assertion
2. **C Eval replay UI** — 拉高 demo 觀感
3. **B6 CI cold-cache** — 觸發即知
4. **D 系列** — 依預算與研究目標決定

---

## 何時更新本文件

- 完成任一項時，直接從本文件刪除（紀錄留在 git 歷史 / commit message）
- 發現新的 known issue 時補進對應分類
- 修補方向有更具體計畫時，把 bullet 換成 link 到 commit / PR

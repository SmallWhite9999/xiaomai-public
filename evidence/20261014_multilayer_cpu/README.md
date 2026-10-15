# 2026-10-14 多層 FP32 CPU 盲測：原失敗保留

這份白名單公開資料保存 round 2 的數值證據。**14 案全數未通過固定逐元素門檻**，未證明失敗根因，也未放寬門檻。

| 項目 | 原回條結果 |
| --- | --- |
| 案例 | 14：0 通過、14 失敗 |
| 主比較角色 | 1,066：365 個角色含超標元素 |
| 報告比較元素 | 3,712,272 |
| 超標元素 | 1,776 |
| 最差誤差／允許界限 | 39.65447831742038 倍 |
| G 分段重算原始位元一致性 | 14/14 通過 |
| 共用負控制 | 104/112 通過，8 項失敗 |

門檻固定為 `abs(candidate-reference) <= 8e-6 + 5e-4*abs(reference)`。環境為 Windows、Python 3.12.14、Torch 2.14.0+cpu，每個模型程序單執行緒。模型尺寸為 hidden 64、intermediate 128、vocab 97，使用合成輸入；4 層序列長度 7／64／65／130／257，8 層長度 17／65，各含 weak／strong decay 案例。候選為 FP32，數值參照為 FP64。

- `summary.json`：執行範圍、總數、負控制與未完成事項。
- `cases.json`：14 案逐案統計、最差角色與實際報告座標。
- `role_metrics.json`：1,066 個角色的判定、元素數、最差座標與值；**沒有完整輸出／梯度 tensor**。
- `SOURCE_PROVENANCE.json`：來源快照、原回條 pins 與公開 JSON 雜湊。

G 的位元一致性與完整協定 verdict 分開記錄。因共用負控制未全過，原完整 G 協定欄仍為 FAIL；不能把 14/14 位元相同改寫成整體通過。

4 個 noncontiguous 案例各有 2 個負控制失敗。原注入可能寫到 reshape 副本；後續修補有 6 項合成測試通過，**尚未套用原執行，也不是新的模型驗收**。公開資料保留原 104/112 與整體 FAIL。

來源快照完成時間為 `2026-10-14T15:01:42.656371+00:00`，來源提交為 `b60ffa98f406f396a14fd9a2d5f0a19f7459c2e8`；整理日為 2026-10-15。原快照中的 `publication_of_this_round=false` 是當時的歷史欄位；不拿它當本次公開發布的狀態，也不修改歷史原件。

從倉庫根目錄執行：

```console
python examples/read_admission_summary.py
python -m unittest discover -s tests -p test_public_evidence.py -v
```

示範只用 Python 標準庫：讀取公開 JSON、核對 hash、重算每個**報告最差座標**的差值與允許界限，並將角色統計加總至案例和總表。示範輸出的 `public_numeric_record_check: PASS` 僅代表資料一致，旁邊保留 `scientific_multilayer_FP32_admission: FAIL`。

完整 tensor、原凍結 runner 與核心原始碼維持私有。讀者無法重新搜尋完整 tensor 的最大誤差、獨立重數超標元素，亦無法從這份資料重跑模型。負控制和位元一致性數量是回條報告值，這個示範未重新執行那些控制。沒有新的 GPU 執行、True 接線、更新／保存／冷還原或能力收益驗收。

雜湊辨識 byte 內容，不認證作者。若 provenance 與資料被一起替換，仍需透過可信的版本歷史與來源稽核確認身分。

# 小麥 Xiaomai

**小麥是一個開發中的 AI 研究系統，以「可解釋、可探索的自主推理進化」為目標：它以不完美的開源模型為起點，逐步讓自己的運算、權重、執行狀態與學習機制成為能讀取、檢查與介入的研究對象；探索未知世界與自身，從矛盾形成問題，提出新的概念、表示或方法，以推導、反例和介入實驗檢驗，依結果分岔或回溯，保存有效改變，再檢查新知識與能力能否在重啟與未見情境中持續起作用。**

研究方向與專案整合：**郭俊逸**。完整小麥工作以 ORNITH（Ornith-1.5-9B）開源模型為基礎，逐步把部分研究運算及更新流程接成可追查、可修改、可恢復的程式。

**目前已完成的工程起點：**10/3 的 True CPU 研究分支完成兩次新更新，每次有 25 份研究參數張量改變，並完成保存、新程序接續及該批最新保存點的獨立恢復核對。10/7–10/9，限定研究更新已在凍結 9B 的 GPU 特徵上完成，CPU 研究狀態經冷恢復核對；指定保存點也在兩個新程序中接續下一筆原 credit 研究交易（見 [2026 年 10 月進度](docs/progress_2026-10.md)）。本 repo 另外提供可重跑的小型運算追蹤與保存接續示範。Python 改寫、追蹤與保存接續是讓研究可以被檢查的手段，不是研究目標本身。

**自主問題重定義、新機制創造及持久能力提升，仍待驗證。**完整模型也尚未完成全 Python 轉換；這裡的可解釋性是把說明接到指定運算與改變證據，不代表所有內部機制已可見。

## 研究紀錄

| 文件 | 內容 |
|---|---|
| [雙語週報 / Bilingual updates](updates/README.md) | 研究敘事、技術紀錄與來源格式 / Research notes, technical records and sources. |
| [首期 2026-W41 / First issue](updates/2026-W41.md) | 募資前紀錄，研究證據截至 10/9 / Pre-campaign record; research evidence through October 9. |
| [2026 年 10 月進度](docs/progress_2026-10.md) | R10–R19 與 10/9 停點：真 9B 研究更新、保存接續、算子改寫，以及仍未通過的項目。 |
| [定位與前人工作](docs/positioning.md) | 和 FunSearch、AlphaEvolve、SEAL、ROME／MEMIT、DGM 等工作的逐項比較，以及小麥還缺哪些證據。 |
| [實驗紀錄](research/experiments.md) | 問題形成循環 v1–v5，以及本機 GPU 上 53 次請求的經驗與記憶對照：分母、結果與限制。 |
| [系統設計概念](research/system_design.md) | 驗證板與有限度自我演化、研究循環的概念圖，以及全專案共用的證據原則。 |
| [工程追查紀錄](research/engineering.md) | A100 上的記憶體不足、工具解析、推論快取、保存重載四個案例：怎麼找到原因、怎麼證明。 |

## 先看什麼

這個 repo 提供兩個使用公開合成資料的小型 CPU 實驗，以及目前真實模型工作的證據摘要。

| 入口 | 可以核對什麼 |
|---|---|
| [運算追蹤程式](src/xiaomai_trace/chunk_trace.py) | 記錄受限 chunk 計算圖的前向數值與反向依賴；獨立 reader 逐項重算，檢查記錄中的數值與梯度是否一致。 |
| [運算追蹤示範](examples/run_trace_demo.py) | 開啟紀錄後結果是否改變，以及紀錄能否獨立讀回。篡改與拒絕案例由[trace 測試](tests/)另外核對。 |
| [更新與保存示範](examples/run_lifecycle_demo.py) | 以原 PythonTape／PythonAdamW 更新小型合成模型，保存後由新程序恢復，對照下一步輸出、更新、optimizer 與隨機狀態。forward 仍用原生運算，此示範不載入 9B 模型。 |
| [目前進度](docs/current_status.md) | 截至 10/9 的 R13–R19 與六項界線，以及 10/3 公開示範與 True 模型歷史快照。 |
| [下一個能力實驗](docs/next_experiment.md) | 如何檢驗「自行重新定義問題」與「新能力在新問題中繼續有效」。這份計畫尚未執行。 |
| [架構](docs/architecture.md)與[個人貢獻](docs/contributions.md) | 各元件如何接續、哪些依據開源來源，以及郭俊逸負責的研究方向與專案分工。 |

## 為什麼做

這個方向起於 VeriTrade 的模型需要適應新環境、找出失效原因與修正方法。如果人類的提問含錯誤假設或漏掉限制，AI 即使照題完成，也可能得不到真正需要的結果。小麥希望在保留原始目的與必要限制的條件下，自行質疑問題、提出新表示或機制，驗證後繼續探索。這種能力希望能用在不同領域。

可解釋 AI、自我修正、自主研究與自我修改已有前人工作。小麥承接這些方向，檢驗把問題重定義接到可觀察、可撤回的機制改變，能否累積可重用能力。需要與相近方法做正式對照，才能判斷這種連接是否帶來額外作用；目前沒有證據宣稱首創或優於既有方法。

## 執行

在獨立的 Python 3.12 環境安裝依賴，於 repo 根目錄執行：

```console
python -m pip install -e ".[test]"
python examples/run_trace_demo.py --out runs/trace-demo
python -m pytest -q
```

每次示範使用新的輸出目錄。完成後先讀 `runs/trace-demo/RESULT.json`，再看同一目錄的計算圖與接受收據。

第二個示範需要額外安裝 lifecycle 依賴：

```console
python -m pip install -e ".[lifecycle,test]"
python -B examples/run_lifecycle_demo.py --out runs/lifecycle-demo
python -B examples/run_lifecycle_tests.py --out runs/lifecycle-tests
```

摘要寫入 `runs/lifecycle-demo/PUBLIC_SUMMARY.json`。來源預設為 `vendor/lifecycle_source/`，記錄層級預設 `calls`，可用 `--detail lines` 改為行級。監督上限固定為 60 秒，包含新程序恢復；逾時保留失敗原因並清理子程序。這個實驗使用 1,342 個合成模型參數，其中 412 個可訓練；成功只支持這個小型案例的更新與保存接續。

| 已整理的證據 | 來源與用途 |
|---|---|
| [本次公開 trace 示範](evidence/public_trace_20261003/RESULT.json)及[合成計算圖](evidence/public_trace_20261003/COMPLETE_SCALAR_GRAPH.jsonl) | 從本 repo 搬移後入口重新執行的小型合成 CPU 結果與可讀回的逐項記錄。 |
| [本次公開 lifecycle 示範](evidence/public_lifecycle_20261003/PUBLIC_SUMMARY.json)及[測試摘要](evidence/public_lifecycle_20261003/PUBLIC_TEST_SUMMARY.json) | 小型合成更新、保存與新程序接續的公開入口結果；對應[測試程式](examples/run_lifecycle_tests.py)。 |
| [True CPU 工程驗收摘要](evidence/true_cpu_20261003/FINAL_ACCEPTANCE.public.json) | 既有私人 run 的整理摘要，支持指定工程結論；沒有在公開示範中重跑，也不能只用公開 repo 獨立重現完整流程。 |

10/3 的執行範圍與依賴環境見[CHECKS.json](evidence/CHECKS.json)。trace 的 26 項測試及 lifecycle 的 145 項測試分開記錄，均為公開合成 CPU 的歷史快照；不計入 True 能力驗證，也不是 10/11 的新執行。

## 已走到哪裡

2026-10-03 的 True CPU 整合紀錄完成兩次研究分支更新：更新後保存、結束程序，再由新程序恢復並重新計算梯度、更新和保存；最新保存點也另以新程序恢復核對。這些工作建立了「改變能保存、能接續、能檢查」的工程基礎。

上述 10/3 公開示範的通過結果只代表其列出的合成運算與保存範圍；該批次沒有完成 GPU 驗收。10/7–10/9 的限定 A100 研究更新與保存接續，另見[目前進度](docs/current_status.md)與[十月進度](docs/progress_2026-10.md)。完整模型仍有原生 Torch 與外部依賴；各版本及各 run 的證據範圍分開判讀，網站中較早的接續結果不取代本 repo 的日期與收據。

## 來源與授權

沿用程式的來源、整理前後 hash 與改動見[打包來源記錄](provenance/PACKAGING.json)；True 公開摘要的來源與輸出對應見[證據匯出記錄](provenance/TRUE_EVIDENCE_EXPORTS.json)。授權與原 NOTICE 的入口見[LICENSES/README.md](LICENSES/README.md)。此整理尚未新增全域開源授權，沿用來源保留各自授權；不能把沿用程式或模型權重都歸為本專案原創。這份 repo 沒有包含模型權重或私人保存點。

網站：[xiaomailab.com](https://xiaomailab.com/) · 研究方向與專案整合：郭俊逸

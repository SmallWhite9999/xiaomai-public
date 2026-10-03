# 目前進度與證據範圍

更新日期：2026-10-03。

小麥目前已建立部分運算追蹤、參數更新與保存接續的工程基礎。完整自主重新定義問題、創造新機制及持續能力提升，尚未完成能力驗證。

## 這個 repo 可以重跑的內容

公開示範使用合成數值與小型 CPU 案例，讓審查者直接讀程式、執行並檢查結果，不需要真實 9B 模型權重或私人 True 保存點。

| 項目 | 能支持的結論 | 不能由此推論 |
|---|---|---|
| chunk 運算追蹤與獨立 reader | 指定圖的數值、局部導數、反向累加、序列及來源綁定可檢查。 | 全模型已可解釋，或已知道答案錯誤的語義根因。 |
| 開／關紀錄比較 | 本次 journal 的觀察行為沒有改變被比較的結果。 | 所有外部 observer 或所有模型執行都無副作用。 |
| 紀錄破壞與拒絕案例 | 指定篡改、缺漏或不完整紀錄會被拒絕。 | 能偵測一切可能攻擊或未列出的錯誤。 |
| 小型更新與新程序恢復 | 1,342 個合成參數、412 個可訓練；PythonTape／PythonAdamW 更新與新程序恢復後的下一步可按契約核對。forward 仍使用原生運算。 | 9B 模型自主進化、GPU 正確性或跨問題能力提高。 |

本次整理後，已從公開包的入口重新執行並核對。以下依[CHECKS.json](../evidence/CHECKS.json)及對應收據記錄，兩組測試分開計算。

| 整理後實際執行 | 結果與證據 |
|---|---|
| trace 示範 | 合成 CPU B1/T3/H1/K2/V2；1,482 筆事件、542 個節點、926 條反向依賴邊。開／關紀錄的輸出、狀態及六種輸入梯度相同，獨立 reader 已讀回核對。[結果](../evidence/public_trace_20261003/RESULT.json)、[合成計算圖](../evidence/public_trace_20261003/COMPLETE_SCALAR_GRAPH.jsonl)。 |
| trace 測試 | 26 項通過，0 failure/error/skip，包含指定篡改與拒絕案例。從 repo 根目錄執行 `python -m pytest -q`。 |
| lifecycle 示範 | 1,342 個合成參數、412 個可訓練；本次耗時約 40.17 秒，新程序的下一步輸出、更新、optimizer 與 RNG 對照相同。[PUBLIC_SUMMARY.json](../evidence/public_lifecycle_20261003/PUBLIC_SUMMARY.json)。 |
| lifecycle 原來源測試 | 145 項通過，0 failure/error/skip；本次耗時約 52.06 秒。從 repo 根目錄執行 `python -B examples/run_lifecycle_tests.py --out runs/lifecycle-tests`。[測試摘要](../evidence/public_lifecycle_20261003/PUBLIC_TEST_SUMMARY.json)。 |

已驗證的環境為 Windows、Python 3.12.14。lifecycle 復用已確認的 Torch 等套件，這不等於已驗證從乾淨網路安裝全部 lifecycle 依賴。trace 的獨立安裝與整包檢查範圍，以 `CHECKS.json` 的最新記錄為準。上述通過結果仍只支持指定合成 CPU 工程流程，沒有載入私人 True 保存點，也沒有驗證自主研究能力。

## 既有 True 模型工作

以下依 2026-10-03 的 `DELIVERY_FINAL_zh.md` 與 `FINAL_ACCEPTANCE.json` 摘要。它們與公開小型示範的測試分開計算。

| 已核對項目 | 實際範圍 |
|---|---|
| 兩次新梯度計算及參數更新 | 每次有 25 份研究參數張量改變；沒有把已記錄梯度重放成新計算。 |
| 保存後由新程序接續 | 第一筆更新保存後結束程序；新程序恢復、重新觀察工具、重算梯度並完成下一筆更新及保存。 |
| 跨程序狀態核對 | 父保存前與子恢復後的 47 組參數／master、29 組 moments、群組與修訂計數相同。 |
| 最新保存點獨立恢復 | 190 個已材料化 tensor 的 typed bytes、owner aliases 與 CPU Python／NumPy／Torch 隨機狀態已核對；752 個凍結 ancestor refs 保持 lazy。此恢復檢查沒有再做 forward 或更新。 |
| 主要合成回歸 | 247 個不重複主要案例、17 個 subtests 通過；這組回歸本身沒有載入 True。 |
| 實際維度的 chunk 候選 | 合成 CPU FP32 B1/T64/H1/K128/V128，output 與 final-state 共用 loss，六種輸入梯度通過 oracle 容差對照。它沒有接進上述兩筆 True worker。 |

研究梯度 backend 是 CPU FP32 PythonTape，optimizer 是帶可撤回 scalar rounding 的 PythonAdamW；凍結底座特徵與 policy sampling 仍走原生 Torch CPU。

上述 True 結果是既有私人 run 與獨立核對的摘要，並非本 repo 公開 demo 的執行結果。[FINAL_ACCEPTANCE.public.json](../evidence/true_cpu_20261003/FINAL_ACCEPTANCE.public.json)保留工程關鍵標記與未完成範圍；[TRUE_EVIDENCE_EXPORTS.json](../provenance/TRUE_EVIDENCE_EXPORTS.json)說明來源及公開匯出對應。hash 能支持來源與檔案核對，不能單獨證明自主能力。

重跑完整 True 流程仍需要未公開的固定底座、保存點、來源及本機依賴，不能只下載這份 repo 就宣告完整模型冷恢復。網站較早材料與此處 2026-10-03 的 run 分開判讀；這次沒有做兩個新程序從同一狀態執行「同一下一步」的逐位元對照。

## 保留的失敗與未完成項目

第一個 900 秒上限的 True 更新嘗試逾時，保留為失敗；後續明示增加上限才完成。不能因程序被清理，就推論當時曾完成數值回滾。flat-index `core.mm` 的診斷有 25 項測試與數值相等核對，但小案例約 1.71 → 1.66 秒，沒有明顯加速證據，該候選也沒有部署。

目前沒有完成：

- 全部 9B 模型轉成 Python，或模型完整理解自身運算的驗證。
- 完整 32-head、BF16 chunk、GPU 或 True 真模型 chunk 路由驗收。
- 無外部底座依賴的完整冷恢復。
- 兩個新程序從同一狀態執行「同一下一步」的逐位元對照。
- 系統自主發現問題定義錯誤、提出並驗證新機制。
- 新能力在未提示的新題中持續有效，或跨領域遷移。

接下來的實驗應把最後兩項拆成可觀察的判定條件，並加入可以推翻主張的對照。見[下一個能力實驗](next_experiment.md)。

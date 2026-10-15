# 公開示範：快速開始

三個入口由輕到重排列。所有指令都從 **repo 根目錄**執行，使用 Python 3.12；運算追蹤與 lifecycle 建議放在獨立的 Python 環境。帶有 `--out` 的示範每次都要指定**尚未存在的新輸出目錄**；重跑時請換名稱。

| 想先核對什麼 | 入口 | 需要安裝套件嗎？ |
| --- | --- | --- |
| 已公開實驗的數字、判定與加總 | [證據讀回](#1-證據讀回無額外依賴) | 不需要，僅用標準函式庫 |
| 合成小圖的運算與反向依賴 | [運算追蹤](#2-運算追蹤合成小圖) | NumPy；測試另需 pytest |
| 小型模型更新後保存、新程序接續 | [更新與保存](#3-更新與保存選用示範) | 需要額外 lifecycle 依賴 |

## 1. 證據讀回：無額外依賴

```console
python examples/read_admission_summary.py
```

[程式](read_admission_summary.py)讀取 [R20 公開摘要](../evidence/20261013_r20/README.md)與[多層 CPU 盲測資料](../evidence/20261014_multilayer_cpu/README.md)，核對公開檔案雜湊、已報告座標的差值與固定門檻，再加總角色、案例及總表。

**結果直接印在終端機，不產生新檔案。**正常讀回時，輸出會同時包含：

```json
{
  "public_numeric_record_check": "PASS",
  "model_rerun": false,
  "full_tensor_recomparison": false,
  "multilayer_cpu": {
    "case_pass": 0,
    "case_fail": 14,
    "scientific_multilayer_FP32_admission": "FAIL"
  }
}
```

上面只摘錄部分欄位。`PASS` 代表公開資料的算術與統計一致；14 案原模型數值驗收仍是 `FAIL`。這個入口不跑模型，也沒有完整 tensor 可重新搜尋最大誤差或獨立重數超標元素。

若要檢查 reader 對竄改、缺列與不完整資料的拒絕行為，同樣不需額外套件：

```console
python -m unittest discover -s tests -p test_public_evidence.py -v
```

## 2. 運算追蹤：合成小圖

```console
python -m pip install -e ".[test]"
python examples/run_trace_demo.py --out runs/trace-demo
python -m pytest -q
```

[示範](run_trace_demo.py)使用固定的公開合成輸入，記錄受限 chunk 計算圖的前向數值與反向依賴，核對開啟／關閉紀錄的輸出、狀態和梯度位元一致，再由獨立 reader 讀回計算圖。`pytest` 執行 repo 的公開 reader／trace 測試；lifecycle 的原測試需用下一節的獨立入口。

完成後依序查看：

| 輸出 | 內容 |
| --- | --- |
| `runs/trace-demo/RESULT.json` | 判定、獨立讀回結果、開關紀錄的位元比較與來源雜湊 |
| `runs/trace-demo/COMPLETE_SCALAR_GRAPH.jsonl` | 可讀回的逐項運算紀錄 |
| `runs/trace-demo/COMPLETE_SCALAR_GRAPH.ACCEPTED.json` | 限定數值範圍的接受收據 |

成功只支持這份合成 CPU 小圖的記錄與數值檢查；不是完整模型追蹤或 GPU 指令追蹤。失敗時保留的 `.INCOMPLETE`／`.FAILED.json` 不構成接受收據。

## 3. 更新與保存：選用示範

這個入口需要額外依賴；repo 的 `[lifecycle]` 固定 Torch 2.14.0、NumPy 2.5.3、Transformers 5.12.1、Numba 0.67.0、psutil 7.2.2，測試另需 pytest。安裝與執行：

```console
python -m pip install -e ".[lifecycle,test]"
python -B examples/run_lifecycle_demo.py --out runs/lifecycle-demo
python -B examples/run_lifecycle_tests.py --out runs/lifecycle-tests
```

[更新與保存示範](run_lifecycle_demo.py)使用 **1,342 個合成模型參數，其中 412 個可訓練**，以原 PythonTape／PythonAdamW 更新、保存，再由新程序恢復，對照下一步輸出、更新、optimizer 與隨機狀態。前向仍用原生運算；不載入 9B 模型。[測試入口](run_lifecycle_tests.py)另檢查梯度、中斷更新及恢復拒絕案例。

來源預設為 `vendor/lifecycle_source/`，記錄層級預設 `calls`；需要行級紀錄時，在示範命令加入 `--detail lines`。兩個入口各有固定 **60 秒監督上限**，示範的上限包含新程序恢復；逾時回報失敗並清理子程序，請查看保留的日誌。腳本停用模型網路下載與 CUDA 可見裝置。

| 輸出 | 內容 |
| --- | --- |
| `runs/lifecycle-demo/PUBLIC_SUMMARY.json` | 優先閱讀的公開摘要：參數規模、比較、保存接續與限制 |
| `runs/lifecycle-demo/RESULT.json` | 該次示範的原結果 |
| `runs/lifecycle-demo.stdout.txt`、`runs/lifecycle-demo.stderr.txt` | 示範子程序日誌，位於輸出目錄旁 |
| `runs/lifecycle-tests/PUBLIC_TEST_SUMMARY.json` | 該次測試數、失敗、錯誤、略過與耗時 |
| `runs/lifecycle-tests/tests.xml`、`stdout.txt`、`stderr.txt` | 測試明細與日誌，均位於測試輸出目錄內 |

成功只支持這個小型合成 CPU 案例的更新、保存與新程序接續。

## 歷史結果與判讀範圍

不執行也能先看已保存的 [trace 結果與計算圖](../evidence/public_trace_20261003/RESULT.json)、[lifecycle 摘要](../evidence/public_lifecycle_20261003/PUBLIC_SUMMARY.json)及[測試摘要](../evidence/public_lifecycle_20261003/PUBLIC_TEST_SUMMARY.json)。[2026-10-03 CHECKS](../evidence/CHECKS.json)記錄 trace **26 項**及 lifecycle **145 項**測試，兩者分開計算。當時環境為 Windows／Python 3.12.14；lifecycle 沿用已驗證的本機依賴，並非全新網路安裝驗證。trace 另有乾淨安裝檢查。

**145 項是 10/3 的歷史合成 CPU 結果，本次整理未重跑 lifecycle。**目前 repo 的公開 reader／trace 測試通過，也不能改寫成這 145 項已重新通過或完整模型已通過。[True CPU 工程摘要](../evidence/true_cpu_20261003/FINAL_ACCEPTANCE.public.json)則來自既有私人 run，沒有在這些示範中重跑，公開 repo 也不足以獨立重現其完整流程。

三個入口分別檢查公開證據、合成小圖及小型更新保存。其成功均不證明完整 True 模型、正式 BF16、完整 GPU 訓練、自主研究或能力收益；各項真實實驗請依[目前進度](../docs/current_status.md)與[證據索引](../docs/evidence_guide.md)的日期、收據及限制判讀。完整核心原始碼、模型權重與私人保存點維持私有；來源沿用與授權見 [LICENSES](../LICENSES/README.md)。

[返回專案首頁](../README.md)

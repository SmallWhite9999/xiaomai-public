# 個人貢獻與來源

小麥由郭俊逸提出研究方向，目標是製作能追查自身改變、持續改進方法的 AI。這個方向來自 VeriTrade 在環境改變時需要定位模型失效原因與修正方法，並延伸到跨領域的問題形成、實驗與能力保存。

這份公開 repo 把可檢查的 CPU 工程核心、示範、測試與證據整理成獨立入口。小麥沿用開源模型與原有工程來源，也在開發中使用 AI 協助。這裡的「個人貢獻」指研究方向、專案規劃、整合、實作與驗證工作；不能因此把所有沿用程式或模型權重都歸為個人原創。

## 小麥的實作重點

| 工作 | 審查入口 |
|---|---|
| 把研究目標轉成可判定的問題與工程流程 | [架構](architecture.md)、[下一個能力實驗](next_experiment.md)。 |
| 整理可觀察、可獨立重算的數值路徑 | `src/xiaomai_trace/`、`src/chunk_training/` 與追蹤示範。 |
| 讓實驗能檢查更新、保存與新程序恢復 | 保存示範、相關測試及[目前進度](current_status.md)中的 True 證據摘要。 |
| 把失敗、來源版本與驗收範圍留下 | repo 的 evidence、拒絕案例及未完成項目。 |

上表說明整理與實作方向，不是逐檔作者認定。具體來源與整理改動見[PACKAGING.json](../provenance/PACKAGING.json)；不同來源的授權不因重新整理而消失。

## 相關專案經驗

**VeriTrade** 是串接外匯模型訊號、進出場決策、風險管理與交易紀錄回放的系統，獲第十三屆激發學生創意競賽創業實作組第一名。依郭俊逸本人的製作與分工說明，系統由他完整製作；官方團隊名單的兩位成員不代表程式工作平均分配。這份 repo 沒有提供即時交易能力或獨立驗證的投資績效。

**ZeroSCAM** 是即時通話防詐 App，將語音轉成文字、分析詐騙風險，並提供判斷原因與防騙建議。專案由四人合作，獲國立高雄大學資訊工程學系 115 級學生專題研究成果競賽第二名。郭俊逸提出整體專案製作方向，並協助測試所有功能；不把全專案製作歸給他一人。

這兩段經驗呈現需求到系統、測試與檢查的實作取向。小麥則把問題往模型自身的方法推進：能否發現原本的假設或問題定義不再適用，提出新表示，再留下足以驗證改變的證據。

## 模型與公開程式的來源邊界

完整小麥工作以 ORNITH（Ornith-1.5-9B）開源模型為基礎，延伸既有研究程式。公開 CPU 示範使用合成資料，不含 ORNITH 模型權重、私人 True 保存點或完整模型執行環境。完整模型的底座、研究分支與本 repo 小型示範需要分別說明，避免把一段 CPU 運算圖當成整個模型。

| 來源核對入口 | 說明 |
|---|---|
| [PACKAGING.json](../provenance/PACKAGING.json) | 精選 trace 原來源與整理後 hash；這次打包主要調整 import 與目錄布局，不把搬移視為演算法原創。 |
| [lifecycle SOURCE_MANIFEST.json](../vendor/lifecycle_source/SOURCE_MANIFEST.json) | 公開保存示範所需的固定來源清單。 |
| [RESEARCH_NATIVE.json](../vendor/lifecycle_source/provenance/RESEARCH_NATIVE.json) | lifecycle 原有研究運算與來源綁定。 |
| [TRUE_EVIDENCE_EXPORTS.json](../provenance/TRUE_EVIDENCE_EXPORTS.json) | 既有私人 True run 的公開摘要與來源對應，不等同公開完整模型重現。 |
| [LICENSES/README.md](../LICENSES/README.md) | 沿用來源授權及原 NOTICE 的入口。模型來源的使用條件以其上游文件為準。 |

這些入口用來區分原來源、整合改動與公開示範。True 摘要與公開合成 demo 分開核對，hash 也不能替代能力實驗。

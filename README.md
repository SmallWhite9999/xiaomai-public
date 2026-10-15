# 小麥 Xiaomai

### 可解釋 AI、問題重定義與持續學習

小麥是一項開發中的 AI 研究計畫。目標是讓模型能檢查自己的運算與學習過程，從觀測和矛盾中形成問題、提出方法，再用實驗與反例檢查改變是否有效，能否在重啟與新問題中持續起作用。

這個方向起於 VeriTrade 對模型適應與失效原因的追問。研究方向與專案整合：**郭俊逸**。

[專案網站](https://xiaomailab.com/) · [研究總覽](research/README.md) · [目前進度](docs/current_status.md) · [公開示範](examples/README.md)

## 從這裡開始

| 想了解什麼 | 閱讀入口 |
|---|---|
| 小麥在研究哪些問題 | [研究方向與紀錄](research/README.md) |
| 目前做到了哪裡 | [最新狀態](docs/current_status.md) · [研究週報](updates/README.md) |
| 親自試跑公開內容 | [三個小型示範](examples/README.md) |
| 查數字、來源與驗收範圍 | [證據導覽](docs/evidence_guide.md) |

## 目前進度

研究證據截至 **2026-10-14**。

- **更新與保存：**限定研究分支已有參數更新、保存與新程序接續的工程證據。
- **模型區塊：**單一 MLP 的真權重轉 FP32、人工輸入 CUDA 前向與明確反向比較通過。
- **多層整合：**14 個新種子 CPU 案例全部未通過固定 FP32 門檻，數值偏差仍待處理。

自主問題重定義、持久能力提升與跨新題效益，仍待驗證。詳細數字與限制集中在[目前進度](docs/current_status.md)和[最新一期週報](updates/2026-W42.md)。

## 公開範圍

這裡公開研究成果、實驗證據與小型示範；完整核心原始碼、權重與私人保存點維持私有。

[專案與架構說明](docs/README.md) · [公開範圍](docs/publication_scope.md) · [來源與授權](LICENSES/README.md)

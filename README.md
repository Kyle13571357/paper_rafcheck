# paper-refcheck

檢索與引用稽核系統。輸入一段已寫好的文字,逐條 claim 回查它引用的原文,把對不上的地方連同證據推到人眼前。

系統**不判定對錯**,它只負責把可疑處與出處並排放好。

**通用工具,不是綁死這 13 篇論文的腳本。** 語料是 `Corpus` 物件(`corpus.py`),每個模組都吃這個物件而不是寫死路徑;哪些論文、哪個是 tier-0、集合查詢該回傳什麼,都在 `corpus.yaml` / `checks.yaml` 裡宣告,不在程式碼裡。這裡的 12 篇 memory-tiering SOTA 論文 + 1 篇 survey,是驗證用的測試集,不是設計目標。

```bash
python3 corpus.py init ~/papers --out ~/my-review \
  --bibliography draft.docx --tier0-pdf draft.pdf
```

讀資料夾裡每篇 PDF,用最大字級抽標題、正則抓 venue/year,解析 `draft.docx` 的參考文獻列表把 `[9]` 這種標號配回實際檔案,寫出一份可審閱的 `corpus.yaml`——這是本來要花一個下午手工做的事。實測對這個真實草稿(12 篇論文、docx 參考文獻)**12/12 全部配對正確**,而且工具自己認出資料夾裡有一篇論文(`goodsurvey.pdf`,主題其實是 cache partitioning)從未被引用,自動標成「未配對」而不是硬塞。

換一批論文、換一份 survey,跑同一條 pipeline 即可;`checks.yaml` 裡的驗收項目(哪些系統該出現在某個集合查詢裡)也是宣告式的,不用改程式碼。

---

## 動機:一個動工前就存在的真實 finding

本專案要稽核的 survey 中,`AOL` 出現兩個定義:

- §2.2 —  `AOL = Latency / MLP`,標給 Soar/Alto [4]
- §3.1 —  `AOL = L_loaded / (1 + α·(MLP−1))`,TierLab(作者自己的模擬器)實作

兩份 LLM 產出的規劃文件都把第二式寫成「原論文的公式」。實際查 Soar/Alto (OSDI'25) p4 原文:

> we define AOL = Latency / MLP

第一式才是原文定義。而 α=1 時 `1+1·(MLP−1) = MLP`,第二式精確退化成第一式——所以兩者不矛盾,是**未言明的推廣**。

正解不是「A 對 B 錯」,而是回報:同一符號兩處定義不同、關係未說明、其中一處是作者自身實作而非原論文定義。這需要第五類判定 `underspecified`。

---

## 不能弄錯的四件事

1. **單一模型、單一索引。** `ask` 與 `check` 用同一個模型、同一個索引,差別只在 query 怎麼構造、檢索範圍怎麼限定。**沒有任何 fine-tune。**
2. **check 的檢索必須有 scope。** 無引用的 claim 走全域檢索時**強制排除 tier-0**,否則會拿 survey 自己的敘述驗證 survey 自己,系統看似運作正常但結果無意義。
3. **表格不進向量庫。** 「哪些系統支援 2 MB」要的是完整集合;語意相似度只會回傳最像的幾筆,然後靜靜漏掉其餘的。集合型查詢走 `tables/*.json` 的 filter。
4. **eval ground truth 不可由模型產生。** `eval/eval_set.jsonl` 全部人工從原文核對,`eval/validate_eval_set.py` 會再用程式重新驗證一次。

---

## Pipeline

```
papers/*.pdf
   │
   ├─ parse.py           座標法解析 → blocks.jsonl
   │                     x 分欄、y 分列;表格用第一欄當 row anchor
   ├─ quality_check.py   純規則品質閘門 → quality_report.json
   ├─ vision_verify.py   視覺複查(附通道健康檢查)→ vision_report.jsonl
   ├─ build_index.py     三條路徑:向量+BM25 / 表格 JSON / 公式
   ├─ retrieve.py        Hybrid + reranker + doc_id/tier filter
   │
   ├─ ask.py             問句 → 帶出處的回答
   └─ check.py           段落 → 逐 claim 判定 + 原文 span
```

| 檔案 | 作用 |
|---|---|
| `corpus.py` | `Corpus` 物件 + `init` 子命令。每個模組的語料路徑、doc_id 解析、tier 判斷都經過這裡,程式碼裡不寫死任何一篇論文 |
| `corpus.yaml` | `ref → doc_id → tier → file`。`check.py` 的 reference resolution 完全依賴這張表 |
| `checks.yaml` | 這批語料自己的驗收期望(集合查詢該回傳誰、doc_id filter 該收斂到誰)。換語料就換這份宣告,不改 `build_index.py` / `retrieve.py` |
| `llm.py` | 所有模型呼叫的唯一入口,`LLMProvider` 抽象類別 + 各供應商子類別,供應商可換 |
| `units.py` | 數值正規化(`54 µs` == `0.054 ms`),**純程式,不經模型** |
| `eval/eval_set.jsonl` | 39 題人工標註評估集 |
| `eval/run_baseline.py` | 三組對照實驗 |

`tier` 0 = survey 本身(二手轉述),1 = 原文(一手)。

---

## 兩個實際用途

### (1) 寫 survey 時快速查找

```bash
python3 refcheck.py
```

互動式,模型只載入一次(約 16 秒),之後每次查詢約 0.5 秒。
一次性 CLI 每次都要付 16 秒冷啟動,所以查找一律用這個。

```
> how much does a page migration cost        直接檢索,不需 API key
> /doc m5                                    限定只看 M5
> sparse page word level tracking
> /tier 1                                    只看原文,排除 survey 轉述
> /tables 2mb                                表格集合查詢(完整集合)
> /ask what penalty does TierLab use         生成帶出處的回答(需 key)
```

**檢索不需要 API key。** 多數時候看到段落本身、頁碼與 section 就夠了。

### (2) 寫完後校對

```bash
python3 check.py --file draft.docx
```

直接吃 `.docx` / `.pdf` / `.txt`,不必先另存純文字。流程:切段 → 逐條抽 claim →
解析引用 → 鎖定該篇檢索 → 判定 → 附原文 span。

報告**預設只顯示有問題的**,並依嚴重度排序:

```
contradicted → not_found → condition_mismatch → underspecified
→ possibly_missing_citation → unresolvable_reference
```

`--all` 連通過的一起列,`--limit N` 只跑前 N 段。

單段校對:

```bash
python3 check.py --text "Colloid achieves 1.01–1.76× speedup [3]."
```

---

## 執行

```bash
python3 parse.py                      # PDF → blocks.jsonl
```
```bash
python3 quality_check.py              # 規則閘門,輸出待複查頁清單
```
```bash
python3 build_index.py --selftest     # 建索引 + Module C 驗收
```
```bash
python3 retrieve.py --acceptance      # Module D 驗收
```
```bash
python3 eval/validate_eval_set.py     # 驗證評估集本身
```

以上皆為本地執行,不需 API key。

生成層需要設定 provider(預設 DeepSeek):

```bash
export DEEPSEEK_API_KEY=...
```
```bash
python3 ask.py "What migration penalty does TierLab charge?"
```
```bash
python3 check.py --self-audit         # 拿系統審自己的 survey
```
```bash
python3 eval/run_baseline.py --arms none raw system --repeats 3
```

檢索層(embedding / BM25 / reranker)全部本地離線執行。

### 視覺複查

`vision_verify.py` 需要 vision-capable provider。DeepSeek 無 vision 模型,`llm.py` 會**明確報錯**而非把圖悄悄丟掉:

```bash
REFCHECK_PROVIDER=openai python3 vision_verify.py
```

---

## 通道健康檢查(為什麼需要)

視覺通道可能靜默失敗:影像沒送達,但呼叫成功返回,模型照樣輸出一份看似正常的比對報告。

實測(2026-07-30,同一 prompt 但拿掉 `images` 欄位):

```json
{
  "image_token": "NONE",
  "page_summary": "A table with three columns: System, Profiling, and Objective.",
  "discrepancies": [{"location": "Table 2, row MTM [1], column Profiling",
                     "image_shows": "HW sampling (PEBS)", ...}]
}
```

模型宣稱「影像顯示」某內容——**它從未看到任何影像**。

作法:每次呼叫在圖左上角疊一個隨機 token,要求模型先回報。答不出即判定影像未送達,該頁 `channel_failed_needs_human`,並**丟棄該次所有 finding**。

---

## 現況

Phase 1 的 A–G 已完成並通過各自驗收;H 的程式碼完成,實驗待 API key。詳見 [results.md](results.md)。

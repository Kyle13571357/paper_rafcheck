# 企劃書:論文檢索與引用稽核系統

工作名稱 `paper-refcheck`

---

## 1. Reason

### 問題
CS550 survey 涵蓋的 12 篇論文是 2024–25 年的 OSDI / SOSP / ASPLOS / EuroSys 成果,不在 LLM 訓練資料內。直接詢問這些論文的實驗數據,模型會給出看似合理的假數字,且無出處可查、外部也搜尋不到可靠二手資料。

### 為什麼是 RAG,不是 fine-tune
| 需求 | 為何 fine-tune 不適用 |
|---|---|
| 論文要能持續新增 | 每加一篇就得重訓 |
| 回答必須附頁碼出處 | 權重裡的知識指不出來源 |
| 資料不足時要能拒答 | fine-tune 會讓模型更敢答,不是更誠實 |

fine-tune 教的是行為與格式,不是事實。本專案要的是可追溯的事實,所以是 RAG。

### 為什麼不直接把 13 篇塞進 context
這是必被問到的一題,而且在 n=13 這個規模,質疑是合理的。

1. **Scoped retrieval 無法用 context stuffing 取代。** 驗證一句標註 `[9]` 的 claim 時,必須強制只看 M5。13 篇同時在 context 裡時,沒有機制能限制模型參考範圍——它會採用其他篇,包括 survey 自己對 M5 的轉述。這不是慢一點或貴一點,是做不到。
2. **Span 級可追溯。** 直讀 PDF 給出答案,無法機械化驗證該句來自哪一行。稽核用途的前提就是這個。
3. **規模。** n=13 塞得下,n=100 塞不下。設計要能往上走。

成本與注意力稀釋是附帶效果,不是主論據。

### 兩種用途
**同一個模型、同一個索引**,差別只在 query 怎麼構造、檢索範圍怎麼限定。

- **`ask`** — 輸入問句,輸出帶出處的回答
- **`check`** — 輸入已寫好的段落,逐條 claim 比對原文,標出數值錯誤與條件省略

`check` 是專案主體。`ask` 在檢索層建好後幾乎免費附送。

### 為什麼主體是 check
撰寫 survey 時最常見的失真不是抄錯數字,是把原文「在特定配置下達到 X%」壓縮成「達到 X%」。條件掉了,結論就變質。人工逐條回查很累,但機器比對很適合。

### 定位
- 系統的角色是**把可疑處連同證據推到人眼前**,不自動判定對錯
- 結論寫幻覺的**降低幅度**與殘餘錯誤類型,不宣稱消除

---

## 2. Steps

Phase 1 = A–H。Phase 2 = I。每組末的**驗收**是往下走的條件。

### A. Corpus 建置
- **A1** 收齊 12 篇原文 PDF,連同 survey 共 13 篇
- **A2** 建 `corpus.yaml`:`ref → doc_id → tier → file`。先於程式碼定下來,後面所有模組都依賴它
- **A3** 解析器讀座標不讀文字順序:x 值分欄、y 值分列
- **A4** 區塊分類 prose / table / caption / formula,各帶 bbox 與 page
- **A5** 表格還原:x 分欄 + 第一欄 token 當 row anchor,處理多行 cell
- **A6** 濾除 header / footer / 頁碼
- **驗收**:隨機抽 5 頁與原 PDF 逐字對照,雙欄順序與表格欄位皆正確

### B. 品質閘門
- **B1** 規則檢查:CID 亂碼率、覆蓋率落差、BBox 跨欄重疊、表格欄位數一致性、數值單位正則、段落句中斷裂
- **B2** 視覺複查:對 B1 flag 的頁 **+ 全部含表格頁**,渲染 PNG 送 vision model 逐格比對
- **B3** 視覺複查輸出強制 JSON,`severity` 須含 `cannot_verify`
- **B4** 人工裁決 B2 回報的 discrepancy
- **B5** 公式正規化:U+1D400 區段(Mathematical Alphanumeric Symbols)轉回 ASCII
- **驗收**:輸出品質報告——總頁數、flag 率、複查次數、人工修正數

### C. 索引
- **C1** prose 切 chunk,注入 metadata `{doc_id, tier, section, page, block_type}`
- **C2** prose 建向量索引 + BM25 索引
- **C3** table 轉 JSON 進結構化查詢層,**不進向量庫**
- **C4** formula 存 LaTeX
- **驗收**:「哪些系統支援 2 MB THP」用 filter 查得到完整三筆(MTM、NOMAD、NeoMem),一筆不漏

### D. 檢索層
- **D1** Hybrid:BM25 抓縮寫(PEBS、DCSC、FMAR、CIT、THP)+ dense 抓語意
- **D2** 掛 reranker
- **D3** 支援 `doc_id` 與 `tier` 兩個 filter 參數——**這是 G 組的前置條件**
- **驗收**:同一 query 加不加 doc_id filter,結果集正確收斂

### E. ask 模式
- **E1** prompt 強制附出處,context 無數據時回「資料未提及」,禁止推算補值
- **E2** 回答須標明出處是 tier-0(survey 轉述)或 tier-1(原文)
- **E3** 支援 tier filter:可指定只問原文、或只問 survey 怎麼寫的
- **驗收**:A 類題目能答出正確數值並指到正確 section

### F. 評估集標註
- **F1** A 類(corpus 內有答案)10–15 題,逐題記 ground truth 與出處
- **F2** B 類(corpus 內沒有,正解是拒答)10 題
- **F3** C 類(誘導陷阱,考歸屬錯置)10 題
- **F4** 自我稽核種子:預埋 survey 內已知的四個可抓點
- **F5** 存成 `eval_set.jsonl`:`{id, class, question, ground_truth, source, note}`
- **驗收**:每題 ground truth 都能指到具體 page / section / table cell
- **不可外包給模型**——模型標的 ground truth 拿去評模型是循環

### G. check 模式
- **G1** Claim 抽取:`{subject, metric, value, unit, relation, condition, cited_ref, source_span}`
- **G2** Reference resolution:查 corpus.yaml,`[9] → m5_asplos25`
- **G3** Scope 決策(逐 claim,非逐段落)
- **G4** 四類判定:`supported` / `contradicted` / `not_found` / `condition_mismatch`
- **G5** 輸出強制附原文 span
- **驗收**:C 類題目不被誘導;跑自己的 survey 能抓出 F4 預埋的四點

### H. 對照實驗與結案
- **H1** 三組實測:無檢索 / 未清洗文本的檢索 / 本系統
- **H2** 記錄模型版本、日期、prompt、temperature、重複次數 n≥3
- **H3** 指標:numeric exact-match(主)、B 類 refusal rate、C 類 misattribution rate、reference resolution 正確率、Context Precision / Recall
- **H4** Error analysis:殘餘錯誤分類與成因
- **H5** 寫 `results.md` 與 README
- **驗收**:能講出一句有數字的結論

### I. Phase 2
- **I1** 質性 claim 驗證(非數值)
- **I2** 跨層一致性檢查:同題分別只用 tier-0 與只用 tier-1 作答,不一致即紅旗
- **I3** 人工裁決的視覺化面板
- **I4** corpus 擴充

### 不做
聊天 UI、多輪記憶、自建 embedding、自寫 layout model。基礎設施用現成元件,驗證邏輯自建。

---

## 3. Modules

```
paper-refcheck/
├── corpus.yaml
├── parse.py
├── quality_check.py
├── vision_verify.py
├── build_index.py
├── retrieve.py
├── ask.py
├── check.py
├── eval/
│   ├── eval_set.jsonl
│   └── run_baseline.py
├── tables/*.json
└── results.md
```

### corpus.yaml
文件註冊表。`ref → doc_id → tier → file`。

```yaml
- ref: 0
  doc_id: cs550_survey
  tier: 0
  file: papers/survey.pdf
- ref: 9
  doc_id: m5_asplos25
  tier: 1
  file: papers/m5.pdf
```

`tier` 0 = 自己的 survey(二手轉述),1 = 原文(一手)。`check.py` 的 reference resolution 完全依賴這張表。

### parse.py
PDF → `blocks.jsonl`,每筆 `{doc_id, tier, page, section, block_type, bbox, text}`。

核心是座標法:x 值分欄、y 值分列,表格用「x 分欄 + 第一欄 token 當 row anchor」還原多行 cell。已在 survey 實測 Table 1 十二列全對,並抓到扁平抽取會犯的欄位歸屬錯誤。

**公式同樣走座標,不需 OCR。** survey 中看似亂碼的 AOL 公式,實為 CambriaMath 使用 U+1D400 區段(`𝐴` = U+1D434 MATHEMATICAL ITALIC CAPITAL A),codepoint 合法,只是多數終端顯示成方框。三步即可還原:

- U+1D400 區段正規化回 ASCII
- span `size` 較小者判為下標/上標
- drawing rect 中的細長橫線判為分數線,線上為分子、線下為分母

實測結果:`AOL = L_loaded / (1 + α·(MLP − 1))`,確定性還原,無需視覺模型。

### quality_check.py
輸入 `blocks.jsonl` + PDF page,輸出 flag 清單 `{page, check_name, detail}`。純規則,零成本,目的是把需要視覺複查的頁數壓到最小。

### vision_verify.py
輸入被 flag 的頁 + 全部含表格頁,渲染 PNG 與解析結果一起送 vision model。輸出 `{location, extracted, image_shows, severity}`。`severity` 必須含 `cannot_verify`——不給拒答選項模型就會猜。

**另需 channel-level 健康檢查。** 視覺通道可能靜默失敗(影像未送達但呼叫成功返回),此時模型會在沒看到圖的情況下輸出一份看似正常的比對報告。作法:每次呼叫在圖上疊一個隨機 token,要求模型先回報該 token,答不出即判定影像未送達,該頁退回人工。

### build_index.py
建三條路徑:
- prose → 向量索引 + BM25
- table → JSON 結構化層(**不進向量庫**)
- formula → LaTeX

表格走語意檢索是錯的。「哪些系統支援 2 MB THP」需要完整集合,語意相似度答不出集合。

### retrieve.py
Hybrid(BM25 + dense)+ reranker,並支援 `doc_id` 與 `tier` 兩個 filter。這兩個參數是 `check.py` 能成立的前提。

### ask.py
輸入問句 → 檢索 → 生成。預設全域檢索,可指定 tier。回答須標明出處層級,因為 tier-0 是自己的轉述,可能有誤。

### check.py
專案主體。四步:

1. **Claim 抽取** — `condition` 欄位是必要的,沒有它就偵測不到條件被省略
2. **Reference resolution** — 查 corpus.yaml
3. **Scope 決策**
4. **判定與輸出 span**

**Scope 決策邏輯**(逐 claim,非逐段落。一段話裡三個數字各引不同篇,就跑三次):

```
有 cited_ref
   └─ 鎖定該 doc_id 檢索 → 四類判定

無 cited_ref
   ├─ 屬作者自己的實驗數據 → 跳過,不需驗證
   └─ 疑似轉述他人 → 全域檢索但排除 tier-0
        └─ 若在某篇原文找到相符數值
             → 回報「疑似缺少引用標註」
```

排除 tier-0 是關鍵:若允許檢索到 survey 自己的敘述,就變成拿二手描述驗證二手描述,系統看似運作正常但結果無意義。

**四類判定**:
- `supported` 數值與條件都對上
- `contradicted` 原文有明確不同數值
- `not_found` 原文無此數字(引錯篇,或轉述時自行推導)
- `condition_mismatch` 數值對但條件被省略或改寫 ← 最有價值的一類
- `underspecified` 來源本身歧義(同一符號多重定義、關係未說明)→ 退回人工

二元判定會漏掉後兩類。

### eval/eval_set.jsonl
三類題目:

- **A 類** corpus 內有明確答案。例:TierLab migration penalty = 54 µs(§3.1)
- **B 類** corpus 內沒有但問題合理,正解是拒答。例:Colloid 在 zipf_high_mlp 的 P95(模擬跑的是 policy 不是 system)
- **C 類** 誘導陷阱,考歸屬錯置

C 類種子取自真實發生的 LLM 錯誤:

| 題 | 正解 | 誘導答案 |
|---|---|---|
| Soar/Alto 的 AOL 公式 | 見下方說明,正解是**指出來源歧義** | `L/(1+α(MLP−1))` 並標為「原論文公式」——兩份 LLM 產出的規劃文件都這樣寫 |
| NOMAD 多租戶提升幾 % | 無此數字 | 72% — 實際是 Adaptive Migration **相對** NOMAD 的提升,方向相反 |
| 哪些系統支援 2 MB THP | MTM、NOMAD、NeoMem | 漏答 NeoMem |

**AOL 案例的完整說明**(專案最重要的一題):

survey 中 AOL 出現兩個定義:

- §2.2 C — `AOL = Latency / MLP`,標給 Soar/Alto [4]
- §3.1 — `AOL = L_loaded / (1 + α·(MLP−1))`,TierLab 實作

α=1 時 `1+1·(MLP−1) = MLP`,第二式精確退化成第一式。所以兩者不矛盾,是**未言明的推廣**——原文未寫出這層關係。

因此正解不是「A 對 B 錯」,而是回報:同一符號在兩處定義不同、關係未說明、其中一處為作者自身實作而非原論文定義。這需要第五類判定 `underspecified`,或歸入需人工裁決。

此案例的價值:它是專案尚未動工前就出現的**真實 finding**,而且示範了審查員的正確行為——不是判對錯,是把歧義連同證據推到人眼前。README 開頭可用此案例說明動機非假想。

自我稽核種子(拿系統審自己的 survey):
1. §3.3 寫「Figures 1–4」,實際引用的是 Figure 2–5
2. 結論 #2 的「≈300 accesses」是 54 µs ÷ 170 ns = 317.6 的推導值,非引用值
3. Telescope 在 §2.1 是「90%+ precision/recall on 5 TB at ~0.9% of a single CPU」,§3.4.1 壓縮成「High at TB scale / <1% CPU」→ `condition_mismatch`,且發生在同一份文件內
4. §3.1 的 54 µs 標註 calibrated to M5 [9],需回原文確認該數字存在與適用條件

### eval/run_baseline.py
三組對照,指標以 deterministic 為主。數值需正規化後比對(`54 µs` = `0.054 ms`)。

Baseline 那組必須真的跑,不能想像填寫——現在的模型遇到不熟的論文很可能直接拒答而非編造,若報告照「自信胡編」寫但實測拒答,整個對比會垮。

### results.md
數字 + error analysis。預期殘餘錯誤集中於:跨 chunk 綜合、reference resolution 遇複合引用 `[3, 9]`、condition 抽取 recall 偏低、表格 merged cell。

---

## 4. 不能弄錯的四件事

1. **單一模型、單一索引。** `ask` 與 `check` 的差異僅在 query 如何構造與檢索範圍如何限定,不涉及任何模型微調。README 開頭就要寫明,避免被誤讀成訓練了兩個特化模型。
2. **check 的檢索必須有 scope。** 全域檢索會撈到 survey 自身敘述,形成自我循環。
3. **表格不進向量庫。** 集合型查詢要用 filter,不能靠語意相似度。
4. **eval ground truth 不可由模型產生。** 否則評估本身就是循環。

## 5. 務實取捨
- 關鍵表格可人工結構化(20–30 張,一個下午,準確率 100%)
- 檢索層全本地(embedding、BM25、reranker 皆可離線),生成層再決定用本地模型或 API
- 所有 parsing 腳本留在 repo,pipeline 必須可重現

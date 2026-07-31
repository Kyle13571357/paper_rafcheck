# Results

狀態記錄於 2026-07-30。

**重要區分:** 本文件分成兩部分。第一部分是**已實測**的結果,全部由確定性程式產生,不經任何模型。第二部分是**尚未執行**的模型對照實驗——程式碼完成,等 API key。沒有任何數字是想像填寫的。

---

## Part 1 — 已實測(確定性,無模型)

### Corpus

| 項目 | 數量 |
|---|---|
| 文件 | 13(tier-0 survey ×1,tier-1 原文 ×12) |
| 頁數 | 216 |
| blocks | 8,529 |
| chunks | 1,238 |
| 表格(結構化層,不進向量庫) | 88 |
| 公式 | 11 |

blocks 分佈:prose 7,974 / heading 271 / caption 185 / table 88 / formula 11。

### Module A 驗收 — 解析正確性

- **survey Table 1 十二列全對**,含多行 cell 與跨列換行的 `(policy-agnostic)`。
- **AOL 公式確定性還原**,無需視覺模型:
  `AOL = L_loaded / (1+α∙(MLP−1))`
  三步:U+1D400 區段正規化回 ASCII、span `size` 判上下標、drawing rect 細長橫線判分數線。
- 文字覆蓋率(擷取字元 / PDF 原始字元)中位數 **0.988**、平均 **0.985**;唯一低於 0.80 的頁是一頁僅 154 字元的短頁,差額正是刻意濾除的 running header。

### Module B 驗收 — 品質閘門

| 指標 | 值 |
|---|---|
| 總頁數 | 216 |
| 規則 flag 頁數 | 44(20.4%) |
| 需視覺複查頁數 | 74(34.3%) |

flag 分佈:`misencoded_unit` 97(blocker)、`paragraph_split_mid_sentence` 37(info)、`table_parse_flag` 28、`table_ragged_rows` 7、`table_empty_row` 1。

`bbox_crosses_gutter` **0 次**——這條檢查與 `parse.py` 的分欄邏輯互相獨立,零命中代表分欄決策自洽。

`misencoded_unit` 是後來才加的規則,見下方「單位編碼損毀」一節;加入後複查頁數由 60 升至 74。

### Module C 驗收 — 集合查詢完整性

> 「哪些系統支援 2 MB THP」

```
MTM          granularity='2 MB (THP)'   (cs550_survey p5)
NOMAD        granularity='4 KB / 2 MB'  (cs550_survey p5)
NeoMem       granularity='4 KB / 2 MB'  (cs550_survey p3)
RESULT: PASS — nothing missing
```

關鍵:NeoMem 的欄位寫的是 `4 KB / 2 MB`,字面搜尋 `"2 MB THP"` **只會找到 MTM**。正規化成 `2mb` token 後三筆齊全。這正是表格不進向量庫的理由。

### Module D 驗收 — 檢索範圍收斂

同一 query `"page migration latency cost microseconds"`:

| 條件 | 結果 |
|---|---|
| 無 filter | 命中散佈於 6 個不同 doc |
| `doc_id=m5_asplos25` | 8/8 全部來自 M5 |
| `exclude_tier=0` | 剩餘 tier 僅 `[1]`,survey 完全排除 |
| 表格 filter `2mb` | `['MTM','NOMAD','NeoMem']` |

### Module G — scope 決策(確定性部分)

| 情境 | 決策 |
|---|---|
| `[9]` | 鎖定 `m5_asplos25` |
| `[3, 9]` 複合引用 | 鎖定 `['colloid_sosp24','m5_asplos25']` |
| `[3, 99]` 部分無效 | 鎖定 `colloid_sosp24`,標記 `partially_resolved:[99]` |
| `[99]` 無效 | `unresolvable_reference` |
| 無引用 + 作者自身結果 | `skip` |
| 無引用 + 疑似轉述 | 全域檢索,**`exclude_tier=0`** |

### 最重要的一筆:單位編碼損毀,以及它如何騙過我自己

`check.py` 的 `numeric_prescreen` 不經模型,只做單位正規化後的數值比對:

```
M5          value='54 µs'      → scope ['m5_asplos25']  → NOT FOUND   ← 偽陰性
Colloid     value='1.01-1.76x' → scope ['colloid_sosp24'] → FOUND '1.76' p10
Telescope   value='90%'        → scope ['telescope_atc24'] → FOUND p2
Adaptive    value='72.0%'      → scope ['adaptive_migration_arxiv25'] → FOUND p12
```

第一列是**偽陰性**,而且我一度把它當成 survey 的引用錯誤寫進本文件與評估集。實際上 M5 p12 白紙黑字寫著:

> This is not enough of the number of accesses to amortize the cost of page migration
> (~54µs in our setup), which requires more than 318 accesses (= 54µs/(270ns − 100ns)) on average.

**survey 的引用是正確的。** 錯的是解析與我的查證方法。

**成因:** M5 以 `LibertineMathMI` 排版單位,而該字型的 ToUnicode 表映射錯誤:

| 原文字元 | 文字層實際碼位 |
|---|---|
| `µ` | U+1D44D MATHEMATICAL ITALIC CAPITAL Z |
| `n` | U+1D43F (L) |
| `s` | U+1D440 (M) |

於是 `54 µs` 抽出來是 `54𝑍𝑀`,經 `parse.py` 的 U+1D400 正規化後變成字面乾淨的 **`54ZM`**;`270 ns` 變成 `270LM`。

**為什麼這是最危險的一類:** 文字層解碼**沒有任何錯誤**——沒有 replacement character、沒有 CID 亂碼、正規化後是合法英文字母。編碼檢查抓不到,而任何用「µs」為關鍵字的查詢永遠不會命中。值明明在,看起來正常,卻是隱形的。

**全 corpus 普查:97 處、8/12 篇論文**受影響(MTM `40𝑘𝑈`、M5 `140–170𝐿𝑀`、Chrono `5.5𝐿𝑀𝑁𝑂𝑃`、NeoMem `2.3×2.3𝑚𝑚`⋯)。

**連帶暴露的方法論缺陷:** 我原本的 `validate_eval_set.py` 用**同一條 regex** 去「複查」這個缺席主張,於是複查通過。用同一個方法驗證同一個方法,證明不了任何事。現已改為偵測「數字後接 U+1D400 區段字元」,命中時警告「此類數量的缺席主張必須看渲染頁面,不能看文字層」。

**發現途徑:** 使用者直接看渲染後的 PDF 頁面,一眼看出數字就在那裡。這正好從反面驗證了 B2 視覺複查的設計——**圖像通道能救回文字通道丟失的東西**,而 `misencoded_unit` 現在以 blocker 等級把這 97 處全部送進視覺複查佇列。

修正紀錄:評估集 `C05` / `C08` / `S02` / `S04` 全部反向,新增 `S06` 記錄真正的缺陷(在語料,不在 survey),`B03` 換成真正不存在的問題。

### 稽核過程中發現的原文瑕疵

**Adaptive Migration (ref 5) 自相矛盾。** friendly / unfriendly 標籤在兩處對調:

- p2:migration-**unfriendly** → 14.8%;migration-**friendly** → 36.0%
- p12:14.8% 「with migration-**friendly** workloads」;36.0% 「with migration-**unfriendly** workloads」

p2 才是邏輯自洽的版本(關掉遷移應該幫助「遷移有害」的工作負載)。survey 採用 p2 的說法,是對的。這筆是建評估集時查原文查出來的,已收為 `C06`,正解是**回報歧義**而非選邊。

**Colloid 的條件被省略。** 原文 p11:

> even at the maximum alternate tier unloaded latency (2.7× of default tier), Colloid still achieves 1.01–1.76×, 1.03–1.76× and 1.01–1.63× performance improvement for HeMem, TPP and MEMTIS, respectively

survey §2.2 壓縮成「Achieves 1.01–1.76× speedup」——數值正確,條件(2.7× 的 alternate tier latency、僅 HeMem)全部消失。典型 `condition_mismatch`。

### 視覺通道健康檢查 — 實測驗證其必要性

拿掉 `images` 欄位、其餘完全相同的負向對照:

模型回傳了一份**完整且自信**的報告——包含一句 `page_summary`("A table with three columns: System, Profiling, and Objective")與一筆 discrepancy,其 `image_shows` 欄位聲稱頁面顯示了什麼。**它從未收到任何影像。**

token 檢查抓到了:回報 `NONE` ≠ 預期 token → `channel_ok: false` → `channel_failed_needs_human`,該次所有 finding 丟棄。

這不是假想風險,是實測到的行為。

### 視覺複查試跑(19/60 頁,已停止)

本機 7B 視覺模型跑了 19 頁後停止(每頁 200–2200 秒,機器過熱)。剩餘 41 頁待改用雲端 provider。

| 項目 | 值 |
|---|---|
| 記錄數 | 19 |
| `clean` | 9 |
| `discrepancies_found` | 4 |
| `request_failed` | 6 |
| 通道檢查通過率 | 13/13(所有成功呼叫) |
| 保留的 discrepancy | 6(major 4 / minor 1 / cannot_verify 1) |
| 自我一致而濾除的偽陽性 | 32 |

濾除數(32)遠多於保留數(6):模型很常把「我檢查過而且相符」也填成 discrepancy,`extracted == image_shows` 時直接丟棄。

抓到的真實問題包括:

- `adaptive_migration_arxiv25` p9 **[major]**:Table 2 CXL Memory 頻寬欄位,解析結果是空的 `Read : Write :`,原圖是 `Read : 17.8GB/s, Write : 15.8GB/s`——真實資料遺失。
- `cs550_survey` p12 **[major]**:`Alternate tier` vs 原圖 `Alternate tier saturated`,多行 cell 被切斷。
- `cs550_survey` p8 **[major]**:`Migration budget` 列的欄位順序錯置。

其中兩筆與我人工檢視時獨立發現的問題吻合。

---

## Part 2 — 尚未執行

以下需要 `DEEPSEEK_API_KEY`,程式碼已完成:

- **H1** 三組對照:`none`(無檢索)/ `raw`(未清洗文本檢索)/ `system`(本系統)
- **H2** 記錄 model 版本、日期、prompt、temperature、n≥3
- **H3** 指標:numeric exact-match(主)、B 類 refusal rate、C 類 misattribution rate、reference resolution 正確率、context recall
- **H4** Error analysis

執行:

```bash
export DEEPSEEK_API_KEY=... && python3 eval/run_baseline.py --arms none raw system --repeats 3
```

`raw` 那組刻意做成「最直覺的第一版」:`page.get_text()` 直接切固定長度、無座標分欄、無 filter、無 reranker。有這組才分得出改善來自「有檢索」還是「解析與 scoping」。

**baseline 那組必須真的跑。** 現在的模型遇到不熟的論文很可能直接拒答而非編造;若報告照「自信胡編」的假設寫,而實測是拒答,整個對比會垮。

### 已知限制(會進 error analysis)

1. **Soar/Alto 原文的 AOL 公式抽取不完整。** 原文以堆疊分數呈現(`AOL = Latency` 上,`MLP` 下)且用一般字體,`parse.py` 的公式偵測要求 math 字體或 U+1D400 碼位,因此漏抓,文字層留下 `we define AOL = Latency` 斷句。survey 內的 AOL 公式(CambriaMath)則完整還原。
2. **114 → 98 頁仍偏碎。** 分欄改用 x0 直方圖後大幅改善,殘餘多為圖表座標標籤與參考文獻頁,屬正常現象而非拼接錯誤。
3. **`near_figure_caption_possibly_not_a_table`** 這條 flag 精確度普通:`find_tables()` 會把格狀排版的圖標籤誤判為表格。目前靠 flag + 視覺複查兜底,未自動排除。
4. **視覺複查僅完成 19/74 頁**,且用的是本機 7B 模型;若改用雲端模型補完,須註明兩段方法不同。
5. **單位編碼損毀目前只偵測、未修復。** 97 處已標為 blocker 並送入視覺複查佇列,但 `blocks.jsonl` 內仍是 `54ZM`,因此「54 µs」這類查詢在 M5 上仍會落空。修復需要從渲染頁面取回真值(視覺通道),或建立 per-font 的還原表;前者較穩健,但要等 vision provider 就緒。這是目前已知**最影響檢索正確性**的缺陷。

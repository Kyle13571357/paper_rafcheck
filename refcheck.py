#!/usr/bin/env python3
"""Interactive session for the two things this corpus is actually used for:
looking something up while writing, and proofreading what was written.

Why this exists rather than repeated CLI calls: loading the embedder and the
cross-encoder costs ~17s, and a one-shot command pays that on every question.
Held open, the same query costs ~0.5s. That is the difference between a tool
you consult while writing and one you avoid.

Retrieval needs no API key. Looking a passage up and seeing its page and
section is often the whole answer, so that path works offline; generated
answers and claim-checking light up when a key is set.

    python3 refcheck.py

    > how much does a page migration cost          look it up (no key needed)
    > /ask what migration penalty does TierLab use  generated answer + citations
    > /check <paste a sentence>                     verify claims in that text
    > /doc m5        /tier 1        /k 8            scope the search
    > /tables 2mb                                   set query over tables
    > /help
"""

import re
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
CORPUS_YAML = ROOT / "corpus.yaml"

BANNER = """paper-refcheck  —  查找與校對

  <問題>            檢索原文段落(不需 API key)
  /ask <問題>       生成帶出處的回答（需 API key）
  /check <文字>     逐條 claim 比對原文（需 API key）
  /tables <pattern> 表格集合查詢，例如 /tables 2mb
  /doc <id|off>     限定單篇，例如 /doc m5
  /tier <0|1|off>   0=survey 轉述, 1=原文
  /k <n>            回傳筆數（預設 6）
  /docs             列出 corpus
  /help  /quit
"""


def short_id(corpus, fragment):
    """Accept 'm5' for 'm5_asplos25' -- typing full doc_ids while writing is
    friction the tool does not need to impose."""
    frag = fragment.lower()
    exact = [e["doc_id"] for e in corpus if e["doc_id"].lower() == frag]
    if exact:
        return exact[0]
    hits = [e["doc_id"] for e in corpus
            if frag in e["doc_id"].lower() or frag in (e.get("short") or "").lower()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        print(f"  '{fragment}' 對應多筆: {hits}")
        return None
    print(f"  找不到 '{fragment}'（用 /docs 看清單）")
    return None


def fmt_hit(i, h, corpus_by_id, width=320):
    meta = corpus_by_id.get(h["doc_id"], {})
    pages = (f"p{h['page_start']}" if h["page_start"] == h["page_end"]
             else f"p{h['page_start']}-{h['page_end']}")
    tier = "tier-0 轉述" if h["tier"] == 0 else "tier-1 原文"
    section = h.get("section") or "—"
    text = re.sub(r"\s+", " ", h["text"]).strip()
    if len(text) > width:
        text = text[:width] + "…"
    print(f"\n[{i}] {meta.get('short', h['doc_id'])} · {tier} · {pages}")
    print(f"    §{section}")
    print(f"    {text}")


def main():
    corpus = yaml.safe_load(CORPUS_YAML.read_text())
    corpus_by_id = {e["doc_id"]: e for e in corpus}

    print(BANNER)
    print("載入檢索模型…", end="", flush=True)
    t0 = time.time()
    from retrieve import Retriever, query_tables
    r = Retriever()
    r.search("warmup", k=1)          # force the lazy model loads now, not mid-question
    print(f" 完成（{time.time() - t0:.0f}s，之後每次查詢約 0.5s）\n")

    import llm
    cfg = llm.config()
    has_key = bool(cfg["api_key"]) or cfg["provider"] == "ollama"
    if not has_key:
        print(f"注意：{cfg['key_env']} 未設定 → /ask 與 /check 無法使用，"
              f"檢索功能不受影響。\n")

    state = {"doc": None, "tier": None, "k": 6}

    def scope_label():
        bits = []
        if state["doc"]:
            bits.append(f"doc={state['doc']}")
        if state["tier"] is not None:
            bits.append(f"tier={state['tier']}")
        return f" [{' '.join(bits)}]" if bits else ""

    while True:
        try:
            line = input(f"\n>{scope_label()} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line in ("/quit", "/q", "/exit"):
            break
        if line in ("/help", "/h", "?"):
            print(BANNER)
            continue
        if line == "/docs":
            for e in corpus:
                print(f"  [{e['ref']:>2}] {e['doc_id']:<28} tier {e['tier']}  "
                      f"{e.get('short', '')}")
            continue

        m = re.match(r"^/doc\s+(\S+)$", line)
        if m:
            if m.group(1) in ("off", "none", "-"):
                state["doc"] = None
                print("  已取消 doc 限定")
            else:
                d = short_id(corpus, m.group(1))
                if d:
                    state["doc"] = d
                    print(f"  限定 {d}")
            continue

        m = re.match(r"^/tier\s+(\S+)$", line)
        if m:
            v = m.group(1)
            if v in ("off", "none", "-"):
                state["tier"] = None
                print("  已取消 tier 限定")
            elif v in ("0", "1"):
                state["tier"] = int(v)
                print(f"  限定 tier {v}" + ("（survey 自身轉述）" if v == "0"
                                            else "（原文）"))
            else:
                print("  用法：/tier 0 | /tier 1 | /tier off")
            continue

        m = re.match(r"^/k\s+(\d+)$", line)
        if m:
            state["k"] = max(1, min(20, int(m.group(1))))
            print(f"  k = {state['k']}")
            continue

        m = re.match(r"^/tables\s+(.+)$", line)
        if m:
            rows = query_tables(m.group(1).strip(), doc_id=state["doc"],
                                tier=state["tier"])
            if not rows:
                print("  表格層無相符列")
            for h in rows:
                flag = f"  ⚠ {h['quality_flag']}" if h.get("quality_flag") else ""
                print(f"  {h['doc_id']} p{h['page']} | {h['row_label']} | "
                      f"{h['column']} = {h['value']!r}{flag}")
            print(f"\n  共 {len(rows)} 列（完整集合，非 top-k）")
            continue

        m = re.match(r"^/(ask|check)\s+(.+)$", line, re.S)
        if m:
            verb, payload = m.group(1), m.group(2).strip()
            if not has_key:
                print(f"  需要 {cfg['key_env']}。純檢索請直接輸入問題。")
                continue
            try:
                if verb == "ask":
                    from ask import ask, render
                    render(ask(payload, retriever=r, k=state["k"],
                               doc_id=state["doc"], tier=state["tier"]))
                else:
                    from check import check_passage, render as render_check
                    render_check(check_passage(payload, retriever=r, k=state["k"]))
            except Exception as e:                      # noqa: BLE001
                print(f"  失敗：{type(e).__name__}: {e}")
            continue

        if line.startswith("/"):
            print("  未知指令,/help 看說明")
            continue

        t = time.time()
        hits = r.search(line, k=state["k"], doc_id=state["doc"], tier=state["tier"])
        if not hits:
            print("  沒有命中")
            continue
        for i, h in enumerate(hits, 1):
            fmt_hit(i, h, corpus_by_id)
        print(f"\n  {len(hits)} 筆 · {time.time() - t:.1f}s")

    print("bye")


if __name__ == "__main__":
    main()

"""
MedExACT RAG Pipeline v8 (GPU + Accuracy + Ensemble)
FIXES IN v8:
• HEADERS: Added aliases (HPI, ROS, Discharge Diagnoses, etc.) based on team EDA.
• REGEX: get_section_name and regex_stage now operate on raw_text to preserve \n boundaries.
• INLINE HEADERS: Regex upgraded to capture inline colon-terminated headers.
"""
import argparse, json, logging, pickle, re, string, sys, time
from pathlib import Path
from typing import Optional
import requests

def _ensure(pip_name, import_name=""):
    import importlib
    try:
        importlib.import_module(import_name or pip_name)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "-q", "--break-system-packages"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

_ensure("rich")
_ensure("sentence-transformers", "sentence_transformers")
_ensure("faiss-gpu-cu12", "faiss")
_ensure("numpy<2", "numpy")

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn, TextColumn
import numpy as np

console = Console(highlight=False)
BAR = "━" * 70

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE  = "rag_index.faiss"
CHUNKS_FILE = "rag_chunks.pkl"
EMBED_DIM   = 384
MODEL_PROFILES = {
    "gemma2:9b":       {"chunk_size": 2500, "overlap": 200, "note": "good quality"},
    "llama3.1:8b":     {"chunk_size": 3500, "overlap": 300, "note": "High overlap required"},
    "qwen2.5:14b":     {"chunk_size": 3000, "overlap": 200, "note": "great JSON"},
    "qwen2.5:32b":     {"chunk_size": 4000, "overlap": 300, "note": "JSON MASTER"},
    "gemma2:27b":      {"chunk_size": 6000, "overlap": 300, "note": "A6000 sweet spot"},
    "llama3.1:70b":              {"chunk_size": 8000, "overlap": 400, "note": "A6000 max"},
    "llama3:70b":                {"chunk_size": 8000, "overlap": 400, "note": "A6000 max"},
    # API models via NVIDIA NIM / Gemini
    "moonshotai/kimi-k2-instruct":                   {"chunk_size": 4000, "overlap": 500, "note": "Kimi API"},
    "meta/llama-3.3-70b-instruct":                   {"chunk_size": 4000, "overlap": 500, "note": "Llama API"},
    "mistralai/mistral-large-3-675b-instruct-2512":  {"chunk_size": 4000, "overlap": 500, "note": "Mistral API"},
    "deepseek-ai/deepseek-v3.1":                     {"chunk_size": 4000, "overlap": 500, "note": "DeepSeek API"},
    "qwen/qwen3.5-122b-a10b":                        {"chunk_size": 4000, "overlap": 500, "note": "Qwen API"},
    "gemini-2.5-flash":                              {"chunk_size": 4000, "overlap": 500, "note": "Gemini API"},
}
DEFAULT_PROFILE = {"chunk_size": 1500, "overlap": 150, "note": "safe-fallback"}

VALID_CATEGORIES = {
    "1": "Category 1: Contact related", "2": "Category 2: Gathering information",
    "3": "Category 3: Defining problem", "4": "Category 4: Treatment goal",
    "5": "Category 5: Drug related", "6": "Category 6: Therapeutic procedure related",
    "7": "Category 7: Evaluating test result", "8": "Category 8: Deferment",
    "9": "Category 9: Advice and precaution",
}

CATEGORY_ALIASES = {
    "contact": "1", "gathering": "2", "information": "2", "defining": "3",
    "problem": "3", "diagnosis": "3", "diagnoses": "3", "treatment": "4",
    "goal": "4", "drug": "5", "medication": "5", "medications": "5", "medicine": "5",
    "therapeutic": "6", "procedure": "6", "surgery": "6", "evaluating": "7",
    "test": "7", "result": "7", "lab": "7", "laboratory": "7", "deferment": "8",
    "defer": "8", "hold": "8", "advice": "9", "precaution": "9", "instructions": "9",
    "follow-up": "1", "followup": "1", "appointment": "1", "appointments": "1",
}

SKIP_SECTIONS = {"social history", "family history"}

SECTION_HEADERS = [
    "admission date", "allergies", "chief complaint",
    "major surgical or invasive procedure", "major surgical",
    "history of present illness", "hpi",
    "review of systems", "ros",
    "past medical history", "social history", "family history",
    "physical exam", "physical examination",
    "pertinent results", "imaging", "microbiology",
    "brief hospital course", "transitional issues",
    "medications on admission",
    "discharge medications", "medications on discharge", "medications at the time of discharge",
    "discharge disposition", "facility",
    "discharge diagnosis", "discharge diagnoses", "primary diagnosis", "secondary diagnosis",
    "discharge condition", "discharge instructions", "followup instructions", "followup",
    # Bob's findings — abbreviations (43 HPI, 73+55 ROS, etc.)
    "hpi", "ros", "pmh", "pmhx", "psh", "pshx",
    # 2-letter abbreviations
    "cc", "fh", "sh", "pe", "bhc",
    # Variants from CSV analysis (354 unique sequences)
    "cardiac history", "oncologic history", "past surgical history",
    "psychiatric history", "substance abuse history", "other past history",
    "hospital course", "micu course", "ccu course", "icu course",
    "course on floor", "on transfer to floor", "on discharge",
    "home meds", "meds on transfer", "current medications",
    "primary diagnosis", "secondary diagnosis", "final diagnosis",
    "impression", "assessment", "assessment and plan",
    "microbiology", "micro", "imaging", "labs", "studies",
    "admission labs", "discharge labs", "pertinent labs",
    "allergies", "immunizations", "incisions", "wound care",
    "activity", "diet", "personal care", "general drain care",
    "medication changes", "start", "stop", "changes to your medications",
    "admission exam", "discharge exam", "physical examination",
    "neurologic", "cranial nerves", "motor", "reflexes", "pulses",
    "admission physical", "discharge physical", "discharge pe",
    "on admission", "on discharge", "at discharge",
]

def get_model_profile(model_name: str) -> dict:
    lower = model_name.lower()
    for key, prof in MODEL_PROFILES.items():
        if key in lower:
            return dict(prof)
    return dict(DEFAULT_PROFILE)

def normalize_text(raw: str) -> str: 
    return raw.replace("\n", " ")

def correct_offsets(pred_anns: list, norm_text: str, raw_text: str) -> list:
    for ann in pred_anns:
        text = ann['decision']
        start = ann.get('start_offset', -1)
        end = ann.get('end_offset', -1)
        if start >= 0 and end <= len(norm_text) and norm_text[start:end].lower() == text.lower():
            continue 
        hits, pos, t_low, n_low = [], 0, norm_text.lower(), text.lower()
        while True:
            idx = t_low.find(n_low, pos)
            if idx == -1: break
            hits.append(idx)
            pos = idx + 1
        if hits:
            best_idx = min(hits, key=lambda x: abs(x - max(0, start)))
            ann['start_offset'], ann['end_offset'] = best_idx, best_idx + len(text)
            continue
        clean_text = text.strip(string.punctuation + " ")
        if clean_text and len(clean_text) > 3:
            hits, pos, n_clean = [], 0, clean_text.lower()
            while True:
                idx = t_low.find(n_clean, pos)
                if idx == -1: break
                hits.append(idx)
                pos = idx + 1
            if hits:
                best_idx = min(hits, key=lambda x: abs(x - max(0, start)))
                ann['start_offset'], ann['end_offset'] = best_idx, best_idx + len(clean_text)
                ann['decision'] = norm_text[best_idx:best_idx+len(clean_text)]
    return pred_anns

def make_chunks(text: str, chunk_size: int = 3000, overlap: int = 200) -> list:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            sp = text.rfind(" ", start, end)
            if sp > start + chunk_size // 2: 
                end = sp + 1
        chunks.append((start, text[start:end]))
        if end == len(text): 
            break
        start = end - overlap
    return chunks

def get_section_name(text: str, offset: int) -> str:
    best, best_pos = "", -1
    search_text = text[:offset]
    for h in SECTION_HEADERS:
        pattern = r"(?i)(?:^" + re.escape(h) + r"\s*(?:\n|$)|\b" + re.escape(h) + r"\s*:)"
        for m in re.finditer(pattern, search_text, re.MULTILINE):
            if m.start() > best_pos: 
                best_pos, best = m.start(), h.lower()
    return best

def should_skip_chunk(text: str, chunk_start: int) -> bool: 
    return any(s in get_section_name(text, chunk_start + 50) for s in SKIP_SECTIONS)

def _find_all_ci(text: str, needle: str) -> list:
    hits, pos, t_low, n_low = [], 0, text.lower(), needle.lower()
    while True:
        idx = t_low.find(n_low, pos)
        if idx == -1: 
            break
        hits.append((idx, idx + len(needle)))
        pos = idx + 1
    return hits

def fuzzy_find_offset(norm_text: str, decision: str, chunk_start: int) -> Optional[tuple]:
    dn = decision.replace("\n", " ")
    for v in dict.fromkeys([dn, dn.rstrip(". ,[\t "), dn.strip(), dn.rstrip(". ,[\t ").strip(), re.sub(r"\s+", " ", dn).strip()]):
        if not v: 
            continue
        hits = _find_all_ci(norm_text, v)
        if hits: 
            return min(hits, key=lambda x: abs(x[0] - chunk_start))
    return None

def normalize_category(cat: str) -> Optional[str]:
    if not cat: return None
    cat = cat.strip()
    if cat in VALID_CATEGORIES.values(): return cat
    if (num_match := re.search(r"\b([1-9])\b", cat)): return VALID_CATEGORIES.get(num_match.group(1))
    cat_lower = cat.lower()
    for alias, num in CATEGORY_ALIASES.items():
        if alias in cat_lower: return VALID_CATEGORIES.get(num)
    for valid_cat in VALID_CATEGORIES.values():
        if valid_cat.lower() in cat_lower or cat_lower in valid_cat.lower(): return valid_cat
    return None

def regex_stage(raw_text: str) -> list:
    results = []
    if (fu_m := re.search(r"Followup Instructions|Followup", raw_text, re.IGNORECASE)):
        section = raw_text[fu_m.start():]
        for m in re.finditer(r"(Department:|With:|When:).*?(?=(Department:|With:|When:|Discharge|\Z))", section, re.DOTALL | re.IGNORECASE):
            span = m.group().strip()
            if len(span) > 10:
                abs_s = fu_m.start() + m.start()
                results.append({"decision": span.replace('\n', ' '), "category": "Category 1: Contact related", "start_offset": abs_s, "end_offset": abs_s + len(span), "_source": "regex"})
    
    med_re = re.compile(r"^(?:\d+\.|\*|-)?\s*([A-Za-z][\w\s\-/(),.*%\[\]]{4,80})", re.MULTILINE)
    for sec_name in ["Medications on Admission", "Discharge Medications", "Medications on Discharge", "Medications at the Time of Discharge"]:
        if not (sec_m := re.search(re.escape(sec_name), raw_text, re.IGNORECASE)): continue
        nxt = re.search(r"\s[A-Z][a-z ]+:\s", raw_text[sec_m.end():])
        section = raw_text[sec_m.end():sec_m.end() + (nxt.start() if nxt else len(raw_text) - sec_m.end())]
        for m in med_re.finditer(section):
            span = m.group(1).strip().rstrip(".")
            if len(span) > 5 and not span.isupper():
                abs_s = sec_m.end() + m.start(1)
                results.append({"decision": span.replace('\n', ' '), "category": "Category 5: Drug related", "start_offset": abs_s, "end_offset": abs_s + len(span), "_source": "regex"})
    return results

def merge_predictions(rag_anns: list, base_anns: list) -> list:
    merged = list(rag_anns)
    covered = [(a["start_offset"], a["end_offset"]) for a in rag_anns]
    for bann in base_anns:
        bs, be = bann.get("start_offset", -1), bann.get("end_offset", -1)
        if bs < 0 or be < 0: continue
        conflict = next((i for i, (rs, re_) in enumerate(covered) if max(bs, rs) < min(be, re_)), None)
        if conflict is None:
            merged.append(bann)
            covered.append((bs, be))
        else:
            rag_len = covered[conflict][1] - covered[conflict][0]
            base_len = be - bs
            if base_len < rag_len:
                merged[conflict] = bann
                covered[conflict] = (bs, be)
    merged.sort(key=lambda x: x["start_offset"])
    for i, ann in enumerate(merged): ann["annotation_id"] = f"ENS_{i:04d}"
    return merged

def build_index(train_dir: Path, index_path: Path, chunk_size: int = 2000, overlap: int = 200) -> None:
    import faiss
    from sentence_transformers import SentenceTransformer
    index_path.mkdir(parents=True, exist_ok=True)
    embedder = SentenceTransformer(EMBED_MODEL)
    all_chunks, texts_to_embed = [], []
    for txt_path in sorted(train_dir.glob("*.txt")):
        if not (gold_path := txt_path.with_suffix(".json")).exists(): continue
        raw_text = txt_path.read_text(encoding="utf-8")
        norm_text = normalize_text(raw_text)
        annotations = json.loads(gold_path.read_text(encoding="utf-8")).get("annotations", [])
        for (chunk_start, chunk_text) in make_chunks(norm_text, chunk_size, overlap):
            if should_skip_chunk(raw_text, chunk_start): continue
            chunk_anns = [a for a in annotations if int(a.get("start_offset", -1)) >= 0 and chunk_start <= (int(a.get("start_offset", -1)) + int(a.get("end_offset", -1))) / 2 < chunk_start + len(chunk_text)]
            if chunk_anns:
                all_chunks.append({"chunk_text": chunk_text, "annotations": chunk_anns, "doc_id": txt_path.stem, "chunk_start": chunk_start, "chunk_end": chunk_start + len(chunk_text), "section": get_section_name(raw_text, chunk_start + 50)})
                texts_to_embed.append(chunk_text)
    embeddings = np.vstack([embedder.encode(texts_to_embed[i:i + 2048], show_progress_bar=False, normalize_embeddings=True) for i in range(0, len(texts_to_embed), 2048)]).astype("float32")
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(EMBED_DIM))
    gpu_index.add(embeddings)
    faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), str(index_path / INDEX_FILE))
    with open(index_path / CHUNKS_FILE, "wb") as f: pickle.dump(all_chunks, f)
    console.print(f"[green]✓ Index built: {len(all_chunks)} chunks[/green]")

def retrieve_section_aware(query_emb, index, chunks, top_k, exclude_doc_id, query_section):
    scores, indices = index.search(query_emb.reshape(1, -1), min(top_k * 4, index.ntotal))
    candidates = [(float(s) + (0.1 if query_section and chunks[idx].get("section", "") == query_section else 0.0), chunks[idx]) for s, idx in zip(scores[0], indices[0]) if 0 <= idx < len(chunks) and chunks[idx]["doc_id"] != exclude_doc_id]
    return [c for _, c in sorted(candidates, key=lambda x: x[0], reverse=True)[:top_k]]

RAG_SYSTEM_PROMPT = r"""You are a clinical NLP expert extracting clinical spans from ICU discharge summaries.

TASK: Identify EVERY text span that represents a clinical finding, diagnosis, test result, or actionable medical decision, and classify it into exactly 9 categories. 

CRITICAL STRUCTURE RULE: The document contains standard clinical sections. You MUST process the provided text top-to-bottom. Extract findings like diagnoses (Cat 3) and lab results (Cat 7) just as aggressively as active instructions (Cat 9).

CATEGORIES (MUST use exact format "Category X: Category name"):
Category 1: Contact related (appointments, providers, phone numbers)
Category 2: Gathering information (plans to check/obtain data)
Category 3: Defining problem (diagnoses, symptoms, conditions, signs)
Category 4: Treatment goal (targeted clinical outcomes)
Category 5: Drug related (medications, IVs, doses)
Category 6: Therapeutic procedure related (surgeries, devices, interventions like s/p stenting)
Category 7: Evaluating test result (labs, cultures, imaging outcomes)
Category 8: Deferment (decisions put on hold)
Category 9: Advice and precaution (patient instructions, wound care)

!! CRITICAL RULES FOR EXACT SPAN EXTRACTION !!
1. COPY THE EXACT BOUNDARIES: Do not summarize. Do not trim modifiers. Extract the whole string.
   - WRONG: "CHF"
   - RIGHT: "systolic CHF (EF40-45%), atrial fibrillation"
   - WRONG: "GI bleed"
   - RIGHT: "GI bleed secondary to angiectasias in the duodenum"

2. PRESERVE DE-IDENTIFICATION TOKENS: Copy placeholders character-for-character INCLUDING the asterisks.
   - WRONG: "admitted [Date range (1) 105469] for pneumonia"
   - RIGHT: "admitted [**Date range (1) 105469**] for pneumonia"

3. EXTRACT AGGRESSIVELY: Missing an annotation hurts more than extracting an extra one.

4. NEGATIVE CONSTRAINTS (WHAT NOT TO EXTRACT):
   - Do NOT extract standard nursing tasks or generic hospital routines.
   - Do NOT extract historical narratives that are not active problems.

OUTPUT FORMAT:
Output ONLY a valid JSON array of objects. No markdown, no explanations.
[
  {
    "decision": "EXACT string from text",
    "category": "Category X: Category Name"
  }
]
"""

def build_rag_prompt(chunk_text, retrieved_chunks, section_name, max_anns_per_example=5):
    examples = "\n[EXAMPLES FROM SIMILAR CLINICAL NOTES]\n\n"
    for i, chunk in enumerate(retrieved_chunks, 1):
        ann_json = json.dumps([{"decision": a["decision"], "category": a["category"]} for a in chunk["annotations"][:max_anns_per_example]], indent=2)
        examples += f"Example {i} (doc: {chunk['doc_id']}):\nText: \"{chunk['chunk_text'][:400]}...\"\nAnnotations:\n{ann_json}\n\n"
    context_chunk = f"--- SECTION: {section_name.upper() if section_name else 'CLINICAL NARRATIVE'} ---\n{chunk_text}"
    return RAG_SYSTEM_PROMPT + "\n\n" + examples + "TEXT SEGMENT TO PROCESS:\n" + context_chunk

def call_ollama(model: str, prompt: str, base_url: str, timeout: int, num_ctx: int):
    # Split system prompt from user content for proper role separation
    sep = "\n\nTEXT SEGMENT TO PROCESS:\n"
    if sep in prompt:
        system_part, user_part = prompt.split(sep, 1)
        messages = [{"role": "system", "content": system_part}, {"role": "user", "content": user_part}]
    else:
        messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    for attempt in range(1, 4):
        try:
            r = requests.post(f"{base_url.rstrip('/')}/api/chat", json={"model": model, "messages": messages, "stream": False, "options": {"temperature": 0.0, "num_predict": min(num_ctx // 2, 8192), "num_ctx": num_ctx}}, timeout=timeout)
            r.raise_for_status()
            return r.json()["message"]["content"].strip(), 0
        except Exception:
            if attempt == 3: raise
            time.sleep(5 * attempt)
    return "", 0

def call_api(model: str, prompt: str, base_url: str, api_key: str, timeout: int, api_delay: float):
    """Call NVIDIA NIM / any OpenAI-compatible API endpoint.
    Puts system instructions in the system role and the clinical chunk in the user role.
    For Qwen3.5 and DeepSeek adds /no_think prefix and enable_thinking:False to suppress
    reasoning chains that would break JSON parsing.
    """
    sep = "\n\nTEXT SEGMENT TO PROCESS:\n"
    if sep in prompt:
        system_part, user_part = prompt.split(sep, 1)
    else:
        system_part, user_part = RAG_SYSTEM_PROMPT, prompt

    # Disable thinking mode for models that generate reasoning chains
    no_think_models = ("qwen3", "qwen/qwen3", "deepseek")
    use_no_think = any(k in model.lower() for k in no_think_models)
    if use_no_think:
        user_part = "/no_think\n" + user_part

    payload = {
        "model":       model,
        "messages":    [
            {"role": "system", "content": system_part},
            {"role": "user",   "content": user_part},
        ],
        "temperature": 0.0,
        #"max_tokens":  16384,
        #"max_tokens":8192,
        "max_tokens":4096,
    }
    if use_no_think:
        payload["enable_thinking"] = False

    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    for attempt in range(1, 5):
        try:
            r = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers, json=payload, timeout=timeout,
            )
            
            # --- NEW: Print exact API rejection errors to the terminal ---
            if r.status_code not in (200, 429):
                console.print(f"\n[bold red]API ERROR {r.status_code}: {r.text}[/bold red]")
                
            if r.status_code == 429:
                wait = 60 * attempt
                time.sleep(wait)
                continue
                
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()
            if api_delay > 0:
                time.sleep(api_delay)
            return content, 0
            
        except Exception as e:
            if attempt == 4: raise Exception(f"API Failed after 4 attempts: {e}")
            time.sleep(10 * attempt)
    return "", 0

def parse_llm_output(raw: str) -> list:
    clean = re.sub(r"`(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        p = json.loads(clean)
        if isinstance(p, dict) and "annotations" in p: return [i for i in p["annotations"] if i.get("decision") and i.get("category")]
        if isinstance(p, list): return [i for i in p if isinstance(i, dict) and i.get("decision") and i.get("category")]
    except json.JSONDecodeError: pass
    if (m := re.search(r"\[.*?\]", clean, re.DOTALL)):
        try: return [i for i in json.loads(m.group()) if isinstance(i, dict) and i.get("decision") and i.get("category")]
        except json.JSONDecodeError: pass
    return [{"decision": dm.group(1), "category": cm.group(1)} for line in clean.split("\n") if (dm := re.search(r'"decision"\s*:\s*"([^"]+)"', line)) and (cm := re.search(r'"category"\s*:\s*"([^"]+)"', line))]

def _tokenize(text): return set(re.findall(r"\b\w+\b", text.lower()))
def _span_key(ann): return (int(ann.get("start_offset", -1)), int(ann.get("end_offset", -1)), m.group(1) if (m := re.search(r"\b([1-9])\b", ann.get("category", ""))) else "?")

def evaluate(pred_anns, gold_anns):
    gk, pk = {_span_key(a) for a in gold_anns}, {_span_key(a) for a in pred_anns}
    tp = len(gk & pk)
    sp = tp / len(pk) if pk else 0.0
    sr = tp / len(gk) if gk else 0.0
    sf = 2 * sp * sr / (sp + sr) if (sp + sr) > 0 else 0.0
    pbc = {}
    for a in pred_anns: pbc.setdefault(m.group(1) if (m := re.search(r"\b([1-9])\b", a.get("category", ""))) else "?", []).append(a)
    toks = []
    for g in gold_anns:
        gt = _tokenize(g["decision"]) 
        c = m.group(1) if (m := re.search(r"\b([1-9])\b", g.get("category", ""))) else "?"
        best = 0.0
        for p in pbc.get(c, []):
            pt = _tokenize(p["decision"])
            if (com := gt & pt):
                pr_, re_ = len(com) / len(pt), len(com) / len(gt)
                best = max(best, 2 * pr_ * re_ / (pr_ + re_))
        toks.append(best)
    tf = sum(toks) / len(toks) if toks else 0.0
    return {"span_precision": round(sp, 4), "span_recall": round(sr, 4), "span_f1": round(sf, 4), "token_f1": round(tf, 4), "base_f1": round((sf + tf) / 2, 4), "gold_count": len(gold_anns), "pred_count": len(pred_anns), "tp_span": tp}

def aggregate(all_scores):
    if not all_scores: return {}
    keys = ["span_precision", "span_recall", "span_f1", "token_f1", "base_f1"]
    agg = {k: round(sum(s[k] for s in all_scores) / len(all_scores), 4) for k in keys}
    agg.update({"num_docs": len(all_scores), "total_gold": sum(s["gold_count"] for s in all_scores), "total_pred": sum(s["pred_count"] for s in all_scores), "total_tp": sum(s["tp_span"] for s in all_scores)})
    return agg

def print_eval(doc_id: str, s: dict) -> None:
    w = 24
    bar = lambda v: ("█"*int(v*w)).ljust(w)
    console.print(f"\n  ┌─ [bold]Evaluation:[/bold] {doc_id}\n  │  Gold: {s['gold_count']}  Pred: {s['pred_count']}  TP(span): {s['tp_span']}\n  │  Span  P={s['span_precision']:.3f}  R={s['span_recall']:.3f}  F1={s['span_f1']:.3f}  [green]{bar(s['span_f1'])}[/green]\n  │  Token F1={s['token_f1']:.3f}                         [cyan]{bar(s['token_f1'])}[/cyan]\n  └► [bold]Base F1 = {s['base_f1']:.3f}[/bold]                       [yellow]{bar(s['base_f1'])}[/yellow]\n")

def print_summary(agg: dict) -> None:
    w = 30
    bar = lambda v: ("█"*int(v*w)).ljust(w)
    console.print("\n" + BAR + f"\n  [bold]OVERALL EVALUATION SUMMARY[/bold]\n" + BAR + f"\n  Documents  : {agg['num_docs']}\n  Total Gold : {agg['total_gold']}  Pred : {agg['total_pred']}  TP : {agg['total_tp']}\n  Span  P={agg['span_precision']:.3f}  R={agg['span_recall']:.3f}  F1={agg['span_f1']:.3f}  [green]{bar(agg['span_f1'])}[/green]\n  Token F1   = {agg['token_f1']:.3f}              [cyan]{bar(agg['token_f1'])}[/cyan]\n  [bold]Base F1    = {agg['base_f1']:.3f}[/bold]  (official metric)  [yellow]{bar(agg['base_f1'])}[/yellow]\n" + BAR + "\n")

def load_index(index_path: Path):
    import faiss
    from sentence_transformers import SentenceTransformer
    if not (idx_f := index_path / INDEX_FILE).exists() or not (chk_f := index_path / CHUNKS_FILE).exists():
        console.print(f"  [red]Index not found at {index_path}[/red]")
        sys.exit(1)
    raw_index = faiss.read_index(str(idx_f))
    try:
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, raw_index)
        console.print("  [dim]FAISS: GPU[/dim]")
    except Exception:
        index = raw_index
        console.print("  [dim]FAISS: CPU (no GPU available)[/dim]")
    with open(chk_f, "rb") as f: chunks = pickle.load(f)
    #return index, chunks, SentenceTransformer(EMBED_MODEL)
    return index, chunks, SentenceTransformer(EMBED_MODEL, device="cpu")


def process_document(txt_path, output_dir, args, index, chunks, embedder, doc_index, total_docs, progress, task_id):
    doc_id = txt_path.stem
    out_json = output_dir / f"{args.team_id}-{doc_id}.json"
    raw_text = txt_path.read_text(encoding="utf-8")
    norm_text = normalize_text(raw_text)
    gold_anns = json.loads(txt_path.with_suffix(".json").read_text(encoding="utf-8")).get("annotations", []) if txt_path.with_suffix(".json").exists() else []
    console.print(f"  [dim]◌[/dim]  [{doc_index}/{total_docs}]  [bold]{doc_id}[/bold]")
    all_anns, seen, ann_idx, miss, t0 = [], set(), 0, 0, time.time()

    for item in regex_stage(raw_text):
        if (key := (item["start_offset"], item["end_offset"], item["category"])) not in seen:
            seen.add(key)
            all_anns.append({"decision": item["decision"], "category": item["category"], "start_offset": item["start_offset"], "end_offset": item["end_offset"], "annotation_id": f"{args.team_id}_{ann_idx:04d}"})
            ann_idx += 1

    profile = get_model_profile(args.model)
    chunks_list = make_chunks(norm_text, profile["chunk_size"], profile["overlap"])
    progress.update(task_id, total=len(chunks_list), completed=0)

    for ci, (chunk_start, chunk_text) in enumerate(chunks_list, 1):
        if should_skip_chunk(raw_text, chunk_start): 
            progress.update(task_id, advance=1)
            continue
            
        section_name = get_section_name(raw_text, chunk_start + 50)
        prompt = build_rag_prompt(chunk_text, retrieve_section_aware(embedder.encode(chunk_text, normalize_embeddings=True), index, chunks, args.top_k, doc_id, section_name), section_name)
        
        try:
            # FORCED NVIDIA: Removed if/else to prevent fallback to localhost
            raw, _ = call_api(args.model, prompt, "https://integrate.api.nvidia.com/v1",
                              args.api_key or "", args.timeout,
                              getattr(args, "api_delay", 0.0))
        except Exception as e: 
            console.print(f"\n[bold red]CRITICAL ABORT on chunk {ci} due to API error: {e}[/bold red]")
            raise e
        
        for item in parse_llm_output(raw):
            decision, category = item.get("decision", "").strip(), normalize_category(item.get("category", "").strip())
            if not decision or not category: continue
            if (offsets := fuzzy_find_offset(norm_text, decision, chunk_start)) is None: 
                miss += 1; continue
            if (key := (offsets[0], offsets[1], category)) in seen: continue
            seen.add(key)
            all_anns.append({"decision": decision, "category": category, "start_offset": offsets[0], "end_offset": offsets[1], "annotation_id": f"{args.team_id}_{ann_idx:04d}"})
            ann_idx += 1
        progress.update(task_id, advance=1, description=f"  [dim]⠴[/dim]  chunk {ci}/{len(chunks_list)}   spans={len(all_anns)} miss={miss}")

    all_anns = correct_offsets(all_anns, norm_text, raw_text)
    if args.merge_baseline and (base_path := Path(args.merge_baseline) / f"{args.team_id}-{doc_id}.json").exists(): 
        all_anns = merge_predictions(all_anns, json.loads(base_path.read_text(encoding="utf-8")).get("annotations", []))

    out_json.write_text(json.dumps({"annotator_id": args.team_id, "discharge_summary_id": doc_id, "annotations": all_anns}, indent=4, ensure_ascii=False), encoding="utf-8")
    if gold_anns:
        scores = evaluate(all_anns, gold_anns)
        scores.update({"doc_id": doc_id, "elapsed": round(time.time() - t0, 1)})
        print_eval(doc_id, scores)
        return scores
    return None

def main():
    p = argparse.ArgumentParser()
    subparsers = p.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build-index")
    build_parser.add_argument("--train-dir", required=True)
    build_parser.add_argument("--index-path", required=True)
    build_parser.add_argument("--chunk-size", type=int, default=2000)
    build_parser.add_argument("--overlap", type=int, default=200)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--data-dir", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--index-path", required=True)
    run_parser.add_argument("--model", default="qwen2.5:32b")
    run_parser.add_argument("--team-id", default="team")
    run_parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    run_parser.add_argument("--top-k", type=int, default=5)
    run_parser.add_argument("--num-ctx", type=int, default=16384)
    run_parser.add_argument("--timeout", type=int, default=600)
    run_parser.add_argument("--skip-done", action="store_true")
    run_parser.add_argument("--merge-baseline", default=None)
    run_parser.add_argument("--provider", default="ollama", choices=["ollama", "openai"],
                            help="ollama=local Ollama; openai=NVIDIA NIM or any OpenAI-compatible API")
    run_parser.add_argument("--api-key", default=None, help="API key for openai provider")
    run_parser.add_argument("--api-delay", type=float, default=1.0,
                            help="Seconds to sleep between API calls (rate-limit protection)")
    run_parser.add_argument("--ids", default=None,
                            help="Path to a text file listing doc IDs (one per line) to process")
    args = p.parse_args()

    if args.command == "build-index": 
        build_index(Path(args.train_dir), Path(args.index_path), args.chunk_size, args.overlap)
        return
    data_dir, output_dir, index_path = Path(args.data_dir), Path(args.output_dir), Path(args.index_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    index, chunks, embedder = load_index(index_path)
    # Filter by IDs file if provided
    all_txt = sorted(data_dir.glob("*.txt"))
    if args.ids:
        ids_path = Path(args.ids)
        wanted = set(l.strip() for l in ids_path.read_text().splitlines() if l.strip())
        txt_files = [f for f in all_txt if f.stem in wanted]
        found = len(txt_files)
        missing = len(wanted) - found
        console.print(f"  IDs file: {args.ids} — {len(wanted)} requested, {found} found" +
                      (f", {missing} missing" if missing else ""))
    else:
        txt_files = all_txt
    all_scores = []

    with Progress(SpinnerColumn(spinner_name="dots"), BarColumn(bar_width=22), TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn(), TextColumn("{task.description}"), console=console, transient=True) as progress:
        doc_task = progress.add_task("[bold]Documents[/bold]", total=len(txt_files))
        chunk_task = progress.add_task("chunks", total=1)
        for idx, txt_path in enumerate(txt_files, 1):
            doc_id = txt_path.stem
            progress.update(doc_task, description=f"[dim]{idx}/{len(txt_files)} {doc_id[:20]}[/dim]")
            if args.skip_done and (output_dir / f"{args.team_id}-{doc_id}.json").exists(): 
                progress.advance(doc_task)
                continue
            try:
                if (scores := process_document(txt_path, output_dir, args, index, chunks, embedder, idx, len(txt_files), progress, chunk_task)): 
                    all_scores.append(scores)
            except Exception as e: console.print(f"  [red][FAIL] {doc_id}: {e}[/red]")
            progress.advance(doc_task)
    if (agg := aggregate(all_scores)): print_summary(agg)

if __name__ == "__main__": main()
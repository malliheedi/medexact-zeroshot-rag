"""
MedExACT Batch Zero-Shot Pipeline v13 (A6000 Final Run)
=====================================
FIXES IN v13:
  • HEADERS: Added aliases (HPI, ROS, Discharge Diagnoses, etc.) based on team EDA.
  • REGEX: get_section_name and regex_stage now operate on raw_text to preserve \n boundaries.
  • INLINE HEADERS: Regex upgraded to capture inline colon-terminated headers.
"""
import argparse, json, logging, re, string, sys, time
from pathlib import Path
from typing import Optional
import requests

def _ensure(pip_name, import_name=""):
    import importlib
    try:
        importlib.import_module(import_name or pip_name)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "-q",
            "--break-system-packages"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

_ensure("rich")
from rich.console import Console
from rich.progress import (Progress, SpinnerColumn, BarColumn, TimeElapsedColumn,
    TimeRemainingColumn, TaskProgressColumn, TextColumn)

console = Console(highlight=False)
BAR = "━" * 70

MODEL_PROFILES = {
    "gemma2:9b":       {"chunk_size": 2500, "overlap": 200, "note": "good quality"},
    "llama3.1:8b":     {"chunk_size": 3500, "overlap": 300, "note": "High overlap required"},
    "qwen2.5:14b":     {"chunk_size": 3000, "overlap": 200, "note": "great JSON"},
    "qwen2.5:32b":     {"chunk_size": 4000, "overlap": 300, "note": "JSON MASTER"},
    "gemma2:27b":      {"chunk_size": 6000, "overlap": 300, "note": "A6000 sweet spot"},
    "llama3.1:70b":    {"chunk_size": 8000, "overlap": 400, "note": "A6000 max"},
    "llama3:70b":      {"chunk_size": 8000, "overlap": 400, "note": "A6000 max"},
    # Gemini API — smaller chunks prevent JSON truncation on dense sections
    "gemini-2.5-flash":      {"chunk_size": 1500, "overlap": 150, "note": "Gemini free tier"},
    "gemini-2.5-flash-lite": {"chunk_size": 1500, "overlap": 150, "note": "Gemini free tier"},
    "gemini-2.5-pro":        {"chunk_size": 1500, "overlap": 150, "note": "Gemini Pro"},
    "gemini":                {"chunk_size": 1500, "overlap": 150, "note": "Gemini generic"},
}
DEFAULT_PROFILE = {"chunk_size": 3000, "overlap": 200, "note": "fallback"}

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

# Updated with Team Findings
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
# Bob's findings - Abbreviations (43 HPI, 73+55 ROS, etc.)
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
"on admission", "on discharge", "at discharge"]

def normalize_text(raw: str) -> str:
    return raw.replace("\n", " ")

def correct_offsets(pred_anns: list, norm_text: str, raw_text: str) -> list:
    for ann in pred_anns:
        text = ann['decision']
        start = ann.get('start_offset', -1)
        end = ann.get('end_offset', -1)
        
        # 1. Exact match check
        if start >= 0 and end <= len(norm_text):
            if norm_text[start:end].lower() == text.lower():
                continue 
        
        # 2. Proximity fallback (exact string search)
        hits, pos, t_low, n_low = [], 0, norm_text.lower(), text.lower()
        while True:
            idx = t_low.find(n_low, pos)
            if idx == -1: break
            hits.append(idx)
            pos = idx + 1
            
        if hits:
            best_idx = min(hits, key=lambda x: abs(x - max(0, start)))
            ann['start_offset'] = best_idx
            ann['end_offset'] = best_idx + len(text)
            continue
            
        # 3. Aggressive fallback (stripping hallucinated punctuation)
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
                ann['start_offset'] = best_idx
                ann['end_offset'] = best_idx + len(clean_text)
                # Force exact string match alignment
                ann['decision'] = norm_text[best_idx:best_idx+len(clean_text)]
    return pred_anns

SYSTEM_PROMPT = r"""You are a clinical NLP expert extracting clinical spans from ICU discharge summaries.

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

# Fixed to use raw_text to preserve structural \n anchors
def regex_stage(raw_text: str) -> list:
    results = []
    fu_m = re.search(r"Followup Instructions|Followup", raw_text, re.IGNORECASE)
    if fu_m:
        section = raw_text[fu_m.start():]
        for m in re.finditer(r"(Department:|With:|When:).*?(?=(Department:|With:|When:|Discharge|\Z))", section, re.DOTALL | re.IGNORECASE):
            span = m.group().strip()
            if len(span) > 10:
                abs_s = fu_m.start() + m.start()
                results.append({"decision": span.replace('\n', ' '), "category": "Category 1: Contact related", "start_offset": abs_s, "end_offset": abs_s + len(span), "_source": "regex"})
    
    med_re = re.compile(r"^(?:\d+\.|\*|-)?\s*([A-Za-z][\w\s\-/(),.*%\[\]]{4,80})", re.MULTILINE)
    # Updated array with Team EDA variants
    for sec_name in ["Medications on Admission", "Discharge Medications", "Medications on Discharge", "Medications at the Time of Discharge"]:
        sec_m = re.search(re.escape(sec_name), raw_text, re.IGNORECASE)
        if not sec_m: continue
        nxt = re.search(r"\s[A-Z][a-z ]+:\s", raw_text[sec_m.end():])
        end = sec_m.end() + (nxt.start() if nxt else len(raw_text) - sec_m.end())
        section = raw_text[sec_m.end():end]
        for m in med_re.finditer(section):
            span = m.group(1).strip().rstrip(".")
            if len(span) > 5 and not span.isupper(): 
                abs_s = sec_m.end() + m.start(1)
                results.append({"decision": span.replace('\n', ' '), "category": "Category 5: Drug related", "start_offset": abs_s, "end_offset": abs_s + len(span), "_source": "regex"})
    return results

def _find_all_ci(text: str, needle: str) -> list:
    hits, pos = [], 0
    t_low, n_low = text.lower(), needle.lower()
    while True:
        idx = t_low.find(n_low, pos)
        if idx == -1: break
        hits.append((idx, idx + len(needle)))
        pos = idx + 1
    return hits

def fuzzy_find_offset(norm_text: str, decision: str, chunk_start: int) -> Optional[tuple]:
    dn = decision.replace("\n", " ")
    for v in dict.fromkeys([dn, dn.rstrip(". ,[\t "), dn.strip(), dn.rstrip(". ,[\t ").strip(), re.sub(r"\s+", " ", dn).strip()]):
        if not v: continue
        if hits := _find_all_ci(norm_text, v):
            return min(hits, key=lambda x: abs(x[0] - chunk_start))
    return None

def normalize_category(cat: str) -> Optional[str]:
    if not cat: return None
    cat = cat.strip()
    if cat in VALID_CATEGORIES.values(): return cat
    if num_match := re.search(r"\b([1-9])\b", cat): return VALID_CATEGORIES.get(num_match.group(1))
    cat_lower = cat.lower()
    for alias, num in CATEGORY_ALIASES.items():
        if alias in cat_lower: return VALID_CATEGORIES.get(num)
    for valid_cat in VALID_CATEGORIES.values():
        if valid_cat.lower() in cat_lower or cat_lower in valid_cat.lower(): return valid_cat
    return None

def get_model_profile(model_name: str) -> dict:
    lower = model_name.lower()
    for key, prof in MODEL_PROFILES.items():
        if key in lower: return dict(prof)
    return dict(DEFAULT_PROFILE)

def make_chunks(text: str, chunk_size: int, overlap: int) -> list:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            sp = text.rfind(" ", start, end)
            if sp > start + chunk_size // 2: end = sp + 1
        chunks.append((start, text[start:end]))
        if end == len(text): break
        start = end - overlap
    return chunks

# Fixed to use raw_text with hybrid anchor regex
def get_section_name(text: str, offset: int) -> str:
    best, best_pos = "", -1
    search_text = text[:offset]
    for h in SECTION_HEADERS:
        # Match either inline with a colon OR at the start of a line
        pattern = r"(?i)(?:^" + re.escape(h) + r"\s*(?:\n|$)|\b" + re.escape(h) + r"\s*:)"
        for m in re.finditer(pattern, search_text, re.MULTILINE):
            if m.start() > best_pos: 
                best_pos, best = m.start(), h.lower()
    return best

def should_skip_chunk(text: str, chunk_start: int) -> bool:
    return any(s in get_section_name(text, chunk_start + 50) for s in SKIP_SECTIONS)

def call_ollama(model, segment, base_url, timeout, num_ctx):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": segment}]
    payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0.0, "num_predict": min(num_ctx // 2, 8192), "num_ctx": num_ctx}}
    url = f"{base_url.rstrip('/')}/api/chat"
    for attempt in range(1, 4):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()["message"]["content"].strip(), 0
        except Exception as e:
            if attempt == 3: raise
            time.sleep(5 * attempt)
    return "", 0

def call_api(model, segment, base_url, api_key, timeout):
    """OpenAI-compatible call — works with Gemini, OpenAI, or any compatible endpoint."""
    import requests as _req
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Disable thinking mode for models that support it (Qwen3.5, DeepSeek)
    # /no_think prefix works for Qwen; enable_thinking=False works for vLLM-served models
    user_content = "/no_think\n" + segment

    payload = {
        "model":          model,
        "messages":       [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "temperature":    0.0,
        "max_tokens":     16384,
        "enable_thinking": False,  # Qwen3/3.5 API parameter
    }
    for attempt in range(1, 4):
        try:
            r = _req.post(url, headers=headers, json=payload, timeout=timeout)
            # Handle rate limit (429) with longer backoff
            if r.status_code == 429:
                wait = 60 if attempt == 1 else 120
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            choice = data["choices"][0]
            finish = choice.get("finish_reason", "stop")
            content = choice["message"]["content"].strip()
            if finish == "length":
                # Response was truncated — attempt to salvage partial JSON
                if not content.endswith("]"):
                    content = content + '"]}'  # won't fix it but signals truncation
            return content, 0
        except Exception as e:
            if attempt == 3: raise
            time.sleep(10 * attempt)
    return "", 0


def parse_llm_output(raw: str) -> list:
    clean = re.sub(r"`(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        p = json.loads(clean)
        if isinstance(p, dict) and "annotations" in p: return [i for i in p["annotations"] if i.get("decision") and i.get("category")]
        if isinstance(p, list): return [i for i in p if isinstance(i, dict) and i.get("decision") and i.get("category")]
    except json.JSONDecodeError: pass
    
    # Robust Regex Fallback for truncated JSON
    annotations = []
    matches = re.finditer(r'{\s*"decision"\s*:\s*"([^"]+)"\s*,\s*"category"\s*:\s*"([^"]+)"', clean, re.IGNORECASE)
    for m in matches:
        annotations.append({"decision": m.group(1), "category": m.group(2)})
        
    if not annotations:
        for line in clean.split("\n"):
            dm = re.search(r'"decision"\s*:\s*"([^"]+)"', line)
            cm = re.search(r'"category"\s*:\s*"([^"]+)"', line)
            if dm and cm: annotations.append({"decision": dm.group(1), "category": cm.group(1)})
    return annotations

def _tokenize(text): return set(re.findall(r"\b\w+\b", text.lower()))
def _span_key(ann):
    m = re.search(r"\b([1-9])\b", ann.get("category", ""))
    return (int(ann.get("start_offset", -1)), int(ann.get("end_offset", -1)), m.group(1) if m else "?")

def evaluate(pred_anns, gold_anns):
    gk = {_span_key(a) for a in gold_anns}; pk = {_span_key(a) for a in pred_anns}
    tp = len(gk & pk)
    sp = tp / len(pk) if pk else 0.0
    sr = tp / len(gk) if gk else 0.0
    sf = 2 * sp * sr / (sp + sr) if (sp + sr) > 0 else 0.0
    pbc = {}
    for a in pred_anns:
        m = re.search(r"\b([1-9])\b", a.get("category", ""))
        pbc.setdefault(m.group(1) if m else "?", []).append(a)
    toks = []
    for g in gold_anns:
        m = re.search(r"\b([1-9])\b", g.get("category", ""))
        c = m.group(1) if m else "?"
        gt = _tokenize(g["decision"]); best = 0.0
        for p in pbc.get(c, []):
            pt = _tokenize(p["decision"]); com = gt & pt
            if com:
                pr_ = len(com) / len(pt); re_ = len(com) / len(gt)
                best = max(best, 2 * pr_ * re_ / (pr_ + re_))
        toks.append(best)
    tf = sum(toks) / len(toks) if toks else 0.0
    return {"span_precision": round(sp, 4), "span_recall": round(sr, 4), "span_f1": round(sf, 4), "token_f1": round(tf, 4), "base_f1": round((sf + tf) / 2, 4), "gold_count": len(gold_anns), "pred_count": len(pred_anns), "tp_span": tp}

def aggregate(all_scores):
    if not all_scores: return {}
    keys = ["span_precision", "span_recall", "span_f1", "token_f1", "base_f1"]
    agg = {k: round(sum(s[k] for s in all_scores) / len(all_scores), 4) for k in keys}
    agg.update({"num_docs": len(all_scores), "total_gold": sum(s["gold_count"] for s in all_scores),
        "total_pred": sum(s["pred_count"] for s in all_scores), "total_tp": sum(s["tp_span"] for s in all_scores)})
    return agg

def print_eval(doc_id: str, s: dict) -> None:
    w = 24
    bar = lambda v: ("█"*int(v*w)).ljust(w)
    console.print(f"\n  ┌─ [bold]Evaluation:[/bold] {doc_id}")
    console.print(f"  │  Gold: {s['gold_count']}  Pred: {s['pred_count']}  TP(span): {s['tp_span']}")
    console.print(f"  │  Span  P={s['span_precision']:.3f}  R={s['span_recall']:.3f}  F1={s['span_f1']:.3f}  [green]{bar(s['span_f1'])}[/green]")
    console.print(f"  │  Token F1={s['token_f1']:.3f}                         [cyan]{bar(s['token_f1'])}[/cyan]")
    console.print(f"  └► [bold]Base F1 = {s['base_f1']:.3f}[/bold]                       [yellow]{bar(s['base_f1'])}[/yellow]\n")

def print_summary(agg: dict) -> None:
    w = 30
    bar = lambda v: ("█"*int(v*w)).ljust(w)
    console.print("\n" + BAR + f"\n  [bold]OVERALL EVALUATION SUMMARY[/bold]\n" + BAR)
    console.print(f"  Documents  : {agg['num_docs']}\n  Total Gold : {agg['total_gold']}  Pred : {agg['total_pred']}  TP : {agg['total_tp']}")
    console.print(f"  Span  P={agg['span_precision']:.3f}  R={agg['span_recall']:.3f}  F1={agg['span_f1']:.3f}  [green]{bar(agg['span_f1'])}[/green]")
    console.print(f"  Token F1   = {agg['token_f1']:.3f}              [cyan]{bar(agg['token_f1'])}[/cyan]")
    console.print(f"  [bold]Base F1    = {agg['base_f1']:.3f}[/bold]  (official metric)  [yellow]{bar(agg['base_f1'])}[/yellow]\n" + BAR + "\n")

def process_document(txt_path, output_dir, args, doc_index, total_docs, progress, task_id):
    doc_id = txt_path.stem
    out_json = output_dir / f"{args.team_id}-{doc_id}.json"
    
    log = logging.getLogger(doc_id)
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        fh = logging.FileHandler(output_dir / f"{args.team_id}-{doc_id}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(fh)
    
    raw_text = txt_path.read_text(encoding="utf-8")
    norm_text = normalize_text(raw_text)
    
    gold_anns = json.loads(txt_path.with_suffix(".json").read_text(encoding="utf-8")).get("annotations", []) if txt_path.with_suffix(".json").exists() else []
    console.print(f"  [dim]◌[/dim]  [{doc_index}/{total_docs}]  [bold]{doc_id}[/bold] ({len(norm_text):,} chars | {len(gold_anns)} gold anns)")
    
    all_anns, seen, ann_idx, miss, t0 = [], set(), 0, 0, time.time()
    
    for item in regex_stage(raw_text): # Changed to raw_text
        if (key := (item["start_offset"], item["end_offset"], item["category"])) not in seen:
            seen.add(key)
            all_anns.append({"decision": item["decision"], "category": item["category"], "start_offset": item["start_offset"], "end_offset": item["end_offset"], "annotation_id": f"{args.team_id}_{ann_idx:04d}"})
            ann_idx += 1
    
    profile = get_model_profile(args.model)
    chunks = make_chunks(norm_text, profile["chunk_size"], profile["overlap"])
    progress.update(task_id, total=len(chunks), completed=0)
    
    for ci, (chunk_start, chunk_text) in enumerate(chunks, 1):
        if should_skip_chunk(raw_text, chunk_start): # Changed to raw_text
            progress.update(task_id, advance=1); continue
        
        section_name = get_section_name(raw_text, chunk_start + 50) # Changed to raw_text
        chunk_with_context = f"--- SECTION: {section_name.upper() if section_name else 'CLINICAL NARRATIVE'} ---\n{chunk_text}"
        
        try:
            if getattr(args, 'provider', 'ollama') == 'openai':
                raw, _ = call_api(args.model, chunk_with_context, args.base_url, args.api_key, args.timeout)
                if getattr(args, 'api_delay', 0) > 0:
                    time.sleep(args.api_delay)
            else:
                raw, _ = call_ollama(args.model, chunk_with_context, args.base_url, args.timeout, args.num_ctx)
        except Exception as e:
            log.error(f"chunk {ci} FAILED: {e}"); progress.update(task_id, advance=1); continue
        
        for item in parse_llm_output(raw):
            decision = item.get("decision", "").strip()
            category = normalize_category(item.get("category", "").strip())
            if not decision or not category: continue
            
            offsets = fuzzy_find_offset(norm_text, decision, chunk_start)
            if offsets is None:
                miss += 1; continue
            
            s, e = offsets
            if (key := (s, e, category)) in seen: continue
            seen.add(key)
            all_anns.append({"decision": decision, "category": category, "start_offset": s, "end_offset": e, "annotation_id": f"{args.team_id}_{ann_idx:04d}"})
            ann_idx += 1
        progress.update(task_id, advance=1, description=f"  [dim]⠴[/dim]  chunk {ci}/{len(chunks)}   spans={len(all_anns)} miss={miss}")
    
    all_anns = correct_offsets(all_anns, norm_text, raw_text)
    all_anns.sort(key=lambda x: x["start_offset"])
    
    out_json.write_text(json.dumps({"annotator_id": args.team_id, "discharge_summary_id": doc_id, "annotations": all_anns}, indent=4, ensure_ascii=False), encoding="utf-8")
    
    if gold_anns:
        scores = evaluate(all_anns, gold_anns)
        scores.update({"doc_id": doc_id, "elapsed": round(time.time() - t0, 1)})
        print_eval(doc_id, scores)
        return scores
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="qwen2.5:32b"); p.add_argument("--team-id", default="team")
    p.add_argument("--base-url", default="http://127.0.0.1:11434",
        help="Ollama URL or OpenAI-compatible base URL (e.g. Gemini)")
    p.add_argument("--provider", default="ollama", choices=["ollama","openai"],
        help="ollama (default) or openai (also works for Gemini)")
    p.add_argument("--api-key", default=None,
        help="API key for openai provider (Gemini or OpenAI)")
    p.add_argument("--skip-done", action="store_true"); p.add_argument("--timeout", type=int, default=400)
    p.add_argument("--num-ctx", type=int, default=16384)
    p.add_argument("--ids", default=None,
        help="Path to file with doc IDs to process (e.g. val.txt or test.txt). "
             "If not set, all .txt files in --data-dir are processed.")
    p.add_argument("--api-delay", type=float, default=0.0,
        help="Seconds to wait between API calls (use 6+ for Gemini free tier rate limits)")
    args = p.parse_args()
    
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.ids:
        # Only process the doc IDs listed in the file (e.g. val.txt or test.txt)
        id_list = [l.strip() for l in Path(args.ids).read_text(encoding="utf-8").strip().split("\n") if l.strip()]
        txt_files = [data_dir / f"{doc_id}.txt" for doc_id in id_list]
        txt_files = [f for f in txt_files if f.exists()]
        missing   = [doc_id for doc_id in id_list if not (data_dir / f"{doc_id}.txt").exists()]
        console.print(f"  [dim]IDs file:[/dim] {args.ids} — {len(id_list)} requested, {len(txt_files)} found, {len(missing)} missing")
        if missing:
            console.print(f"  [yellow]Missing:[/yellow] {missing[:5]}{'...' if len(missing)>5 else ''}")
    else:
        txt_files = sorted(data_dir.glob("*.txt"))
    all_scores, success, fail, skipped = [], 0, 0, 0
    
    with Progress(SpinnerColumn(spinner_name="dots"), BarColumn(bar_width=22), TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn(), TextColumn("{task.description}"), console=console, transient=True) as progress:
        doc_task = progress.add_task("[bold]Documents[/bold]", total=len(txt_files))
        chunk_task = progress.add_task("chunks", total=1)
        
        for idx, txt_path in enumerate(txt_files, 1):
            doc_id = txt_path.stem
            progress.update(doc_task, description=f"[dim]{idx}/{len(txt_files)} {doc_id[:20]}[/dim]")
            out_path = output_dir / f"{args.team_id}-{doc_id}.json"
            if args.skip_done and out_path.exists():
                # Skip only if the file has a reasonable number of annotations
                # Small annotation count (<5) means a bad/truncated previous run
                try:
                    prev = json.loads(out_path.read_text(encoding="utf-8"))
                    if len(prev.get("annotations", [])) >= 5:
                        skipped += 1; progress.advance(doc_task); continue
                    else:
                        out_path.unlink()  # delete bad file — will re-run
                except Exception:
                    out_path.unlink()  # corrupted — re-run
            try:
                if scores := process_document(txt_path, output_dir, args, idx, len(txt_files), progress, chunk_task):
                    all_scores.append(scores)
                success += 1
            except Exception as e:
                console.print(f"  [red][FAIL] {doc_id}: {e}[/red]")
                fail += 1
            progress.advance(doc_task)
    
    if agg := aggregate(all_scores):
        print_summary(agg) 
        (output_dir / f"{args.team_id}-eval-summary.json").write_text(json.dumps({"aggregate": agg, "per_doc": all_scores}, indent=4), encoding="utf-8")
    
    console.print(f"\n  Done.  [green]success={success}[/green]  [red]fail={fail}[/red]  [dim]skipped={skipped}[/dim]\n")

if __name__ == "__main__": main()
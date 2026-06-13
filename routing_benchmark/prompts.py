"""LLM prompts + parsers for curating each tier, with template fallbacks (use_llm=False).

Each curate_* returns a payload dict (query/target/gold_route/...) or None to reject the
raw item. build_benchmark.py attaches image/source/meta and assigns ids.
"""

SEG_VERBS = ("segment", "segmentation", "mask", "outline", "highlight", "delineate", "trace out")


def has_seg_verb(text: str) -> bool:
    t = text.lower()
    return any(v in t for v in SEG_VERBS)


# ---------------------------------------------------------------------------
# Tier 1: non-segmentation dialogue (filter VQA)
# ---------------------------------------------------------------------------
_DIALOGUE_SYS = (
    "You label queries for a vision routing benchmark. A query routes to SEGMENTATION if "
    "answering it requires producing a pixel mask or segmenting/localizing a specific object "
    "region; otherwise it routes to DIALOGUE (answered purely with text). "
    "Given a candidate VQA question, decide if it is a CLEAN dialogue query: it must have NO "
    "segmentation/masking/localization intent and be answerable in text. "
    'Output ONLY JSON: {"clean_dialogue": true|false, "reason": "..."}'
)


def curate_dialogue(lm, item, rng, use_llm):
    q = item["text"].strip()
    if not use_llm:
        if has_seg_verb(q):
            return None
        return {"query": q, "target": None, "gold_route": "dialogue", "tier": "dialogue"}
    obj = lm.chat_json(_DIALOGUE_SYS, f"Question: {q}")
    if not obj or not obj.get("clean_dialogue"):
        return None
    if has_seg_verb(q):  # extra guard
        return None
    return {"query": q, "target": None, "gold_route": "dialogue", "tier": "dialogue",
            "llm_meta": {"reason": obj.get("reason")}}


# ---------------------------------------------------------------------------
# Tier 2: explicit segmentation (rephrase a referring expression)
# ---------------------------------------------------------------------------
_EXPLICIT_SYS = (
    "Rewrite a referring expression into a natural, EXPLICIT segmentation instruction that "
    "overtly asks to segment/mask the named target. Vary the phrasing (e.g. 'Segment X.', "
    "'Please output the mask of X.', 'Outline X in the image.'). Keep the target object intact "
    "and do not invent attributes. "
    'Output ONLY JSON: {"query": "...", "target": "..."}'
)


def curate_explicit(lm, item, rng, use_llm):
    expr = item["text"].strip().rstrip(".")
    if not use_llm:
        return {"query": f"Please segment {expr} in this image.", "target": expr,
                "gold_route": "segmentation", "tier": "explicit_seg"}
    obj = lm.chat_json(_EXPLICIT_SYS, f"Referring expression: {expr}")
    query = (obj or {}).get("query", "").strip()
    if not query or not has_seg_verb(query):
        query = f"Please segment {expr} in this image."
    return {"query": query, "target": expr, "gold_route": "segmentation", "tier": "explicit_seg"}


# ---------------------------------------------------------------------------
# Tier 3: reasoning segmentation (verify the query is implicit)
# ---------------------------------------------------------------------------
_REASONING_SYS = (
    "We build a 'reasoning segmentation' tier: the query must REQUIRE segmenting an object "
    "whose identity is only IMPLIED (it must be inferred, not stated by its obvious class "
    "name) and that needs world knowledge or reasoning. "
    "Given a candidate query, judge if it is implicit in this sense. If valid, normalize it so "
    "it clearly asks for a segmentation mask (append a short mask request if missing) WITHOUT "
    "naming the target object. "
    'Output ONLY JSON: {"is_implicit": true|false, "query": "...", "reason": "..."}'
)


def curate_reasoning(lm, item, rng, use_llm):
    q = item["text"].strip()
    if not use_llm:
        query = q if has_seg_verb(q) else (q.rstrip(".") + ". Please output the segmentation mask.")
        return {"query": query, "target": None, "gold_route": "segmentation", "tier": "reasoning_seg"}
    obj = lm.chat_json(_REASONING_SYS, f"Candidate query: {q}")
    if not obj or not obj.get("is_implicit"):
        return None
    query = obj.get("query", "").strip() or q
    if not has_seg_verb(query):
        query = query.rstrip(".") + ". Please output the segmentation mask."
    return {"query": query, "target": None, "gold_route": "segmentation", "tier": "reasoning_seg",
            "llm_meta": {"reason": obj.get("reason")}}


# ---------------------------------------------------------------------------
# Tier 4: mixed-intent (generate dialogue + seg request about one target)
# ---------------------------------------------------------------------------
_MIXED_SYS = (
    "Generate ONE natural user query that blends TWO intents about the SAME target object: "
    "(a) a dialogue question about it (e.g. its color, breed, count, material, function) AND "
    "(b) an explicit request to segment/mask/highlight it. Join them into a single fluent "
    "sentence or two closely linked clauses. "
    'Output ONLY JSON: {"query": "...", "dialogue_part": "...", "seg_part": "..."}'
)


def curate_mixed(lm, item, rng, use_llm):
    expr = item["text"].strip().rstrip(".")
    if not use_llm:
        return {"query": f"What can you tell me about {expr}, and can you also segment it?",
                "target": expr, "gold_route": "segmentation", "tier": "mixed_intent",
                "distractor_route": "dialogue"}
    obj = lm.chat_json(_MIXED_SYS, f"Target object: {expr}\nExample: 'What breed is this dog, and can you also highlight it?'")
    query = (obj or {}).get("query", "").strip()
    if not query or not has_seg_verb(query):
        query = f"What can you tell me about {expr}, and can you also segment it?"
    return {"query": query, "target": expr, "gold_route": "segmentation", "tier": "mixed_intent",
            "distractor_route": "dialogue",
            "llm_meta": {"dialogue_part": (obj or {}).get("dialogue_part"),
                         "seg_part": (obj or {}).get("seg_part")}}


CURATORS = {
    "dialogue": curate_dialogue,
    "explicit_seg": curate_explicit,
    "reasoning_seg": curate_reasoning,
    "mixed_intent": curate_mixed,
}

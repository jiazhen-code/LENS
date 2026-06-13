"""Zero-shot MLLM routing for the difficulty-graded routing benchmark (Table_R 2).

A *frozen* MLLM is prompted to classify each (image, query) into one of four tiers:

    non-seg   : the request does not require segmenting/localizing any object
    explicit  : the target object(s) are named explicitly
    reasoning : the target is not named and must be inferred by reasoning
    mixed     : multiple targets, or explicit + reasoning combined

``trigger = (tier != non-seg)``. From the predicted vs. gold tiers we report per-tier
accuracy, overall accuracy, and the false-/missed-trigger rates.

The router reuses the very same frozen backbone weights the rest of LENS uses (loaded
through the backbone registry), so "frozen MLLM router" means literally the HF checkpoint
named in the model config -- no extra training.
"""
import random
import re

import torch
from PIL import Image


TIERS = ["non-seg", "explicit", "reasoning", "mixed"]
SEG_TIERS = {"explicit", "reasoning", "mixed"}

# Gold-label normalization: map common naming variants onto the 4 canonical tiers.
_TIER_ALIASES = {
    "nonseg": "non-seg", "no-seg": "non-seg", "none": "non-seg", "no": "non-seg",
    "vqa": "non-seg", "non-segmentation": "non-seg", "no-segmentation": "non-seg",
    "general": "non-seg", "qa": "non-seg", "dialogue": "non-seg", "dialog": "non-seg",
    "chat": "non-seg", "conversation": "non-seg",
    "referring": "explicit", "explicit-referring": "explicit", "explicit-seg": "explicit",
    "reason": "reasoning", "implicit": "reasoning", "reasoning-seg": "reasoning",
    "compound": "mixed", "multi": "mixed", "multiple": "mixed", "mixed-seg": "mixed",
}
# Suffixes that merely describe the label and don't change the tier (e.g. "mixed-intent").
_TIER_SUFFIXES = ("-intent", "-tier", "-query", "-type", "-category", "-class", "-level")


def normalize_tier(raw, extra_aliases=None):
    """Map a gold tier label onto one of TIERS, or None if it can't be resolved.

    Handles case, underscores, descriptor suffixes (``mixed-intent`` -> ``mixed``), common
    aliases, and a substring fallback. ``extra_aliases`` (from --tier-map) takes priority.
    """
    t = str(raw).strip().lower().replace("_", "-")
    if extra_aliases and t in extra_aliases:
        return extra_aliases[t]
    if t in TIERS:
        return t
    for suf in _TIER_SUFFIXES:
        if t.endswith(suf) and len(t) > len(suf):
            t = t[: -len(suf)]
            break
    if t in TIERS:
        return t
    if t in _TIER_ALIASES:
        return _TIER_ALIASES[t]
    for canon in TIERS:
        if canon in t:
            return canon
    return None

DEFAULT_ROUTING_PROMPT = ""
# Simple binary router: just decide whether the request needs segmentation at all.
# pred "seg" (yes) => trigger; "non-seg" (no) => don't. This is what the false-/missed-
# trigger rates measure, and matches the benchmark's gold_route (segmentation/dialogue).
BINARY_ROUTING_PROMPT = (
    "Determine whether the user is asking for image segmentation rather than a text-only answer.\n"
    "Answer 'yes' if the request requires identifying and outputting a region/object mask "
    "or locating a specific region/object in the image (e.g., segment, highlight, mark, "
    "point out, locate, outline, provide a mask).\n"
    "Answer 'no' if the request can be answered with text only (e.g., captioning, VQA, "
    "reasoning, counting, attribute recognition).\n"
    "User Input: {question}\n"
    "Answer with only 'yes' or 'no'."
)

BINARY_ROUTING_PROMPT_LLaVA = ('Please judge whether the question "{question}" involves to segment something or highlight areas or call segmentation tool?'
                         "\nAnswer with yes or no.")

# LLaVA-1.5 may pass --system-prompt llava to use it.
LLAVA_SYSTEM_PROMPT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)

# Earliest-match wins; patterns are matched against the lower-cased model output.
_TIER_PATTERNS = {
    "non-seg": r"non-?\s*seg|no\s*segmentation|not?\s*segment|nonseg",
    "explicit": r"explicit",
    "reasoning": r"reason",
    "mixed": r"mixed|multiple targets|both",
}


def parse_binary(text):
    """Map a yes/no answer to 'seg' (needs segmentation) / 'non-seg', or None."""
    t = text.strip().lower()
    y = re.search(r"\byes\b|\byep\b|\byeah\b", t)
    n = re.search(r"\bno\b|\bnope\b", t)
    if y and (not n or y.start() < n.start()):
        return "seg"
    if n and (not y or n.start() < y.start()):
        return "non-seg"
    # fallbacks for free-form answers
    if re.search(r"segment|locali[sz]e|point to|mask", t):
        return "seg"
    if re.search(r"not? (require|need)|just answer|in words|no segmentation|dialogue", t):
        return "non-seg"
    if t[:1] == "y":
        return "seg"
    if t[:1] == "n":
        return "non-seg"
    return None


def parse_tier(text):
    """Map a free-form model answer to one of TIERS, or None if unparseable."""
    t = text.strip().lower().replace("_", "-")
    best, best_pos = None, len(t) + 1
    for tier, pat in _TIER_PATTERNS.items():
        m = re.search(pat, t)
        if m is not None and m.start() < best_pos:
            best, best_pos = tier, m.start()
    return best


def _is_trigger(pred):
    return pred in SEG_TIERS  # None (unparsed) -> not a trigger


def compute_routing_metrics(results):
    """results: list of {"gold": tier, "pred": tier|None}."""
    n = len(results)
    per_tier = {}
    for t in TIERS:
        items = [r for r in results if r["gold"] == t]
        correct = sum(1 for r in items if r["pred"] == t)
        per_tier[t] = {"acc": (correct / len(items)) if items else None,
                       "n": len(items), "correct": correct}
    overall = (sum(1 for r in results if r["pred"] == r["gold"]) / n) if n else None
    nonseg = [r for r in results if r["gold"] == "non-seg"]
    seg = [r for r in results if r["gold"] in SEG_TIERS]
    false_trigger = (sum(1 for r in nonseg if _is_trigger(r["pred"])) / len(nonseg)) if nonseg else None
    missed_trigger = (sum(1 for r in seg if not _is_trigger(r["pred"])) / len(seg)) if seg else None
    unparsed = sum(1 for r in results if r["pred"] is None)
    return {
        "per_tier": per_tier,
        "overall": overall,
        "false_trigger": false_trigger,
        "missed_trigger": missed_trigger,
        "n": n,
        "unparsed": unparsed,
    }


def compute_binary_metrics(results):
    """Binary (seg vs non-seg) routing metrics, same output shape as compute_routing_metrics.

    results: list of {"gold": canonical_tier, "pred": "seg"|"non-seg"|None}. Per-tier 'acc'
    here = fraction of that tier's samples the router routed correctly (non-seg => must NOT
    trigger; explicit/reasoning/mixed => must trigger). overall = trigger accuracy.
    """
    def fired(p):
        return p == "seg"  # None (unparsed) -> treated as no-trigger

    n = len(results)
    per_tier = {}
    for t in TIERS:
        items = [r for r in results if r["gold"] == t]
        gold_fire = t in SEG_TIERS
        correct = sum(1 for r in items if fired(r["pred"]) == gold_fire)
        per_tier[t] = {"acc": (correct / len(items)) if items else None,
                       "n": len(items), "correct": correct}
    overall = (sum(1 for r in results if fired(r["pred"]) == (r["gold"] in SEG_TIERS)) / n) if n else None
    nonseg = [r for r in results if r["gold"] == "non-seg"]
    seg = [r for r in results if r["gold"] in SEG_TIERS]
    false_trigger = (sum(1 for r in nonseg if fired(r["pred"])) / len(nonseg)) if nonseg else None
    missed_trigger = (sum(1 for r in seg if not fired(r["pred"])) / len(seg)) if seg else None
    unparsed = sum(1 for r in results if r["pred"] is None)
    return {
        "per_tier": per_tier,
        "overall": overall,
        "false_trigger": false_trigger,
        "missed_trigger": missed_trigger,
        "n": n,
        "unparsed": unparsed,
    }


class MLLMRouter:
    """Wraps a frozen backbone's HF model + processor and routes a query via generation.

    mode="binary" (default): yes/no -> seg / non-seg (simple, what trigger rates measure).
    mode="tier": 4-way non-seg / explicit / reasoning / mixed.
    """

    def __init__(self, backbone, mode="binary", prompt_template=None,
                 max_new_tokens=8, image_hw=None, system_prompt=None):
        self.backbone = backbone
        self.model = backbone.model
        self.processor = backbone.processor
        self.mode = mode
        if prompt_template is None:
            prompt_template = BINARY_ROUTING_PROMPT if mode == "binary" else DEFAULT_ROUTING_PROMPT
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
        # Optional square resize (default: the MLLM's native preprocessing). The keypoint
        # grid does not matter for routing, so native resolution is usually best.
        self.image_hw = image_hw
        self.system_prompt = system_prompt
        tok = getattr(self.processor, "tokenizer", None)
        self.pad_token_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)

    def render_prompt(self, question):
        """Build the final text string fed to the model (chat-templated, system included)."""
        instruction = self.prompt_template.format(question=question)
        user_msg = {"role": "user", "content": [{"type": "text", "text": instruction}]}
        msgs = ([{"role": "system", "content": self.system_prompt}] if self.system_prompt else []) + [user_msg]
        try:
            text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = None
        # If the template raised OR silently dropped the system role, prepend it ourselves,
        # so the system prompt always actually reaches the model.
        if self.system_prompt and (not text or self.system_prompt not in text):
            base = self.processor.apply_chat_template(
                [user_msg], tokenize=False, add_generation_prompt=True)
            text = self.system_prompt + "\n" + base
        return text

    @torch.no_grad()
    def generate_text(self, image, question):
        # if isinstance(image, str):
        #     image = Image.open(image).convert("RGB")
        # else:
        #     image = image.convert("RGB")
        # if self.image_hw:
        #     image = image.resize((self.image_hw, self.image_hw))

        text = self.render_prompt(question)
        # inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)

        inputs = {
            k: (v.to(self.model.device).to(self.model.dtype)
                if v.is_floating_point() else v.to(self.model.device))
            for k, v in inputs.items() if isinstance(v, torch.Tensor)
        }
        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.pad_token_id,
        )
        new_tokens = gen[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True)

    def predict(self, image, question):
        raw = self.generate_text(image, question)
        pred = parse_binary(raw) if self.mode == "binary" else parse_tier(raw)
        return pred, raw

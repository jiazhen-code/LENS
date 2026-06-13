"""Loader for the difficulty-graded routing benchmark.

Accepts JSONL (one object per line) or a JSON array. Field names are read flexibly so the
loader works with different curation schemas without renaming. For each sample it needs:

  * a question   -- one of: query, question, text, instruction, prompt, sentence, sent, caption
  * an image     -- one of: image_path (preferred, usually absolute), image, img, file, path
  * a tier       -- one of: tier, label, category, class, intent, difficulty, gold_route
                    normalized onto non-seg / explicit / reasoning / mixed
                    (e.g. dialogue -> non-seg, explicit_seg -> explicit, mixed_intent -> mixed)
  * (optional) an id -- one of: id, index, idx, sample_id, uid, qid  (defaults to position)

Pass tier_map (from --tier-map) to override any label that doesn't auto-resolve.
"""

import json
import os

from .router import TIERS, normalize_tier

_QUESTION_KEYS = ("query", "question", "text", "instruction", "prompt", "sentence", "sent", "caption")
_IMAGE_KEYS = ("image_path", "image", "img", "file", "path", "image_file", "filename")
_TIER_KEYS = ("tier", "label", "category", "class", "intent", "difficulty", "gold_tier")
_ID_KEYS = ("id", "index", "idx", "sample_id", "uid", "qid")


def _pick(d, keys):
    for k in keys:
        if k in d and d[k] is not None and str(d[k]).strip() != "":
            return d[k]
    return None


def load_benchmark(path, image_root=None, tier_map=None):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("samples") or data.get("data") or [data]
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]

    items = []
    for i, d in enumerate(data):
        raw_tier = _pick(d, _TIER_KEYS)
        question = _pick(d, _QUESTION_KEYS)
        img = _pick(d, _IMAGE_KEYS)
        if raw_tier is None or question is None or img is None:
            raise ValueError(
                f"sample {i}: could not find required fields. Keys present: {sorted(d.keys())}. "
                f"Need a question {_QUESTION_KEYS}, an image {_IMAGE_KEYS}, and a tier {_TIER_KEYS}."
            )

        tier = normalize_tier(raw_tier, tier_map)
        if tier is None:
            key = str(raw_tier).strip().lower().replace("_", "-")
            raise ValueError(
                f"sample {i}: tier '{raw_tier}' could not be mapped to {TIERS}. "
                f"Pass --tier-map '{{\"{key}\": \"<one of {TIERS}>\"}}'."
            )

        if image_root and not os.path.isabs(str(img)):
            img = os.path.join(image_root, str(img))

        item_id = _pick(d, _ID_KEYS)
        items.append({
            "id": item_id if item_id is not None else i,
            "image": img,
            "question": str(question),
            "tier": tier,
        })
    return items

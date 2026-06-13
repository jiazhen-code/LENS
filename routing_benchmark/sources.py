"""Load raw items from the datasets this repo already uses.

Each loader returns a list of dicts with at least:
    {"text": <raw query/expression>, "image": <path rel to dataset_dir>,
     "image_path": <abs best-effort>, "source": <name>, "meta": {...}}

We over-draw here; the LLM curation in build_benchmark.py filters/rewrites afterwards.
"""

import glob
import json
import os
import sys


def _abs(dataset_dir, rel):
    return os.path.join(dataset_dir, rel)


def _strip_image_token(text: str) -> str:
    return text.replace("<image>", "").replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# VQA / dialogue  (llava_instruct_150k)
# ---------------------------------------------------------------------------
def load_vqa(cfg, rng, n):
    path = _abs(cfg.dataset_dir, cfg.llava_instruct_rel)
    if not os.path.exists(path):
        raise FileNotFoundError(f"llava_instruct json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rng.shuffle(data)

    img_root = _abs(cfg.dataset_dir, cfg.vqa_image_root_rel)
    out = []
    for item in data:
        convs = item.get("conversations", [])
        # first human turn
        q = next((c["value"] for c in convs if c.get("from") == "human"), None)
        if not q:
            continue
        q = _strip_image_token(q)
        if len(q) < 5:
            continue
        img_rel = os.path.join(cfg.vqa_image_root_rel, item["image"])
        out.append({
            "text": q,
            "image": img_rel,
            "image_path": os.path.join(img_root, item["image"]),
            "source": "llava_instruct_150k",
            "meta": {"orig_id": item.get("id")},
        })
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# Explicit segmentation  (refcoco / refcoco+ / refcocog via REFER)
# ---------------------------------------------------------------------------
def load_refcoco(cfg, rng, n):
    # REFER lives in lens/data; make the repo root importable.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from lens.data.refer import REFER
    except Exception as e:  # heavy deps (pycocotools/skimage/matplotlib) or path
        raise ImportError(
            f"Could not import lens.data.refer.REFER ({e}). It needs pycocotools/skimage/"
            f"matplotlib, which are part of the LISA environment."
        )

    data_root = _abs(cfg.dataset_dir, cfg.refcoco_root_rel)
    per_ds = max(1, n // max(1, len(cfg.refcoco_specs)))
    out = []
    for dataset, split_by in cfg.refcoco_specs:
        try:
            refer = REFER(data_root, dataset=dataset, splitBy=split_by)
        except Exception as e:
            print(f"[refcoco] skip {dataset}/{split_by}: {e}")
            continue
        ref_ids = refer.getRefIds(split=cfg.refcoco_split)
        rng.shuffle(ref_ids)
        got = 0
        for rid in ref_ids:
            ref = refer.Refs[rid]
            img = refer.Imgs[ref["image_id"]]
            file_name = img["file_name"]
            sents = [s["sent"].strip() for s in ref.get("sentences", []) if s.get("sent")]
            if not sents:
                continue
            expr = rng.choice(sents)
            img_rel = os.path.join(cfg.refcoco_root_rel, "images/mscoco/images/train2014", file_name)
            out.append({
                "text": expr,
                "image": img_rel,
                "image_path": os.path.join(refer.IMAGE_DIR, file_name),
                "source": dataset,
                "meta": {"ref_id": rid, "category_id": ref.get("category_id")},
            })
            got += 1
            if got >= per_ds:
                break
    rng.shuffle(out)
    return out[:n]


# ---------------------------------------------------------------------------
# Reasoning segmentation  (ReasonSeg)
# ---------------------------------------------------------------------------
def load_reasonseg(cfg, rng, n, sentence_only=True):
    out = []
    for split in cfg.reasonseg_splits:
        pattern = _abs(cfg.dataset_dir, os.path.join("reason_seg", "ReasonSeg", split, "*.json"))
        for json_path in glob.glob(pattern):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    anno = json.load(f)
            except Exception:
                try:
                    with open(json_path, "r", encoding="cp1252") as f:
                        anno = json.load(f)
                except Exception:
                    continue
            texts = anno.get("text", [])
            if isinstance(texts, str):
                texts = [texts]
            is_sentence = bool(anno.get("is_sentence", False))
            # Tier 3 wants the genuinely implicit (sentence) queries.
            if sentence_only and not is_sentence:
                continue
            texts = [t.strip() for t in texts if isinstance(t, str) and len(t.strip()) >= 5]
            if not texts:
                continue
            query = rng.choice(texts)
            img_path = json_path.replace(".json", ".jpg")
            img_rel = os.path.relpath(img_path, cfg.dataset_dir) if img_path.startswith(cfg.dataset_dir) else img_path
            out.append({
                "text": query,
                "image": img_rel,
                "image_path": img_path,
                "source": "ReasonSeg",
                "meta": {"is_sentence": is_sentence, "json": os.path.basename(json_path)},
            })
    rng.shuffle(out)
    return out[:n]


def count_missing_images(samples):
    return sum(1 for s in samples if not os.path.exists(s.get("image_path", "")))

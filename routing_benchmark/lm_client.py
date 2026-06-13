"""Thin client for an lmdeploy-served Qwen3-32B (OpenAI-compatible API).

Used only for *constructing* the benchmark (filtering / rephrasing / generating queries) --
it is the "curated with LLM assistance" piece, not the router under test.

Features: thinking disabled for clean output, <think> stripping as a fallback, bounded
retries, robust JSON extraction, and an on-disk cache so re-runs are cheap and resumable.
"""

import hashlib
import json
import os
import re
import time


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def extract_json(text: str):
    """Best-effort: parse the first JSON object/array in `text`. Returns obj or None."""
    text = strip_think(text)
    # fast path
    try:
        return json.loads(text)
    except Exception:
        pass
    # find the largest balanced {...} or [...] span
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == opener:
                    depth += 1
                elif text[i] == closer:
                    depth -= 1
                    if depth == 0:
                        chunk = text[start:i + 1]
                        try:
                            return json.loads(chunk)
                        except Exception:
                            break
            start = text.find(opener, start + 1)
    return None


class LMClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = cfg.lm_model
        os.makedirs(cfg.cache_dir, exist_ok=True)
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install openai  (used to talk to the lmdeploy API)") from e
        self.client = OpenAI(base_url=cfg.lm_base_url, api_key=cfg.lm_api_key, timeout=cfg.lm_timeout)

    # ---- cache -------------------------------------------------------------
    def _cache_key(self, system, user, temperature):
        h = hashlib.sha256()
        h.update(json.dumps([self.model, system, user, temperature], ensure_ascii=False).encode())
        return h.hexdigest()[:32]

    def _cache_get(self, key):
        p = os.path.join(self.cfg.cache_dir, key + ".json")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)["response"]
            except Exception:
                return None
        return None

    def _cache_put(self, key, system, user, response):
        p = os.path.join(self.cfg.cache_dir, key + ".json")
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"system": system, "user": user, "response": response}, f, ensure_ascii=False)
        except Exception:
            pass

    # ---- calls -------------------------------------------------------------
    def chat(self, system, user, temperature=None):
        temperature = self.cfg.lm_temperature if temperature is None else temperature
        key = self._cache_key(system, user, temperature)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        last_err = None
        for attempt in range(self.cfg.lm_max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=self.cfg.lm_top_p,
                    max_tokens=self.cfg.lm_max_tokens,
                    # Qwen3: turn off thinking for clean, parseable output.
                    extra_body={"chat_template_kwargs": {"enable_thinking": self.cfg.lm_enable_thinking}},
                )
                out = strip_think(resp.choices[0].message.content or "")
                self._cache_put(key, system, user, out)
                return out
            except Exception as e:  # network / server / rate
                last_err = e
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"LM call failed after {self.cfg.lm_max_retries} retries: {last_err}")

    def chat_json(self, system, user, temperature=None):
        """Return a parsed JSON object, or None if the model never produced valid JSON."""
        for attempt in range(self.cfg.lm_max_retries):
            txt = self.chat(system, user, temperature=temperature)
            obj = extract_json(txt)
            if obj is not None:
                return obj
            # nudge: append a stricter instruction and bypass cache by tweaking temperature
            user = user + "\n\nRespond with ONLY a single valid JSON object, no prose."
            temperature = (temperature or self.cfg.lm_temperature) + 0.05 * (attempt + 1)
        return None

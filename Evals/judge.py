"""Local Ollama judge for DeepEval metrics (no hosted-LLM dependency)."""
from __future__ import annotations

import json
import os
import re

import os as _os
_os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "1200")  # local judge is slow

import requests
from deepeval.models import DeepEvalBaseLLM
from dotenv import load_dotenv

load_dotenv()


def _ollama_host() -> str:
    return os.getenv("OLLAMA_URL", "http://localhost:11434").strip().strip('"').strip("'").rstrip("/")


class OllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model: str | None = None, timeout: int = 300):
        self.model_name = model or os.getenv("EVAL_JUDGE_MODEL", "llama3.2")
        self.timeout = timeout

    def load_model(self):
        return self.model_name

    def _call(self, prompt: str, as_json: bool, schema=None) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            # num_predict caps runaway JSON-mode generations; num_ctx keeps the KV cache small
            "options": {"temperature": 0, "num_predict": 2048, "num_ctx": 8192},
        }
        if as_json:
            # Ollama structured outputs: decoding is constrained to the schema
            payload["format"] = schema.model_json_schema() if schema is not None else "json"
        resp = requests.post(f"{_ollama_host()}/api/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("response", "")

    @staticmethod
    def _repair(raw: str):
        """Small models can loop until num_predict cuts the JSON off mid-item.
        Cut back to the last complete item, close the brackets, and drop exact
        duplicate items produced by the loop."""
        def dedupe(v):
            if isinstance(v, list):
                seen, out = set(), []
                for x in v:
                    k = json.dumps(x, sort_keys=True)
                    if k not in seen:
                        seen.add(k)
                        out.append(dedupe(x))
                return out
            if isinstance(v, dict):
                return {k: dedupe(x) for k, x in v.items()}
            return v

        cuts = [m.start() for m in re.finditer(r",\s*[\[{\"]", raw)][::-1][:60]
        for cut in cuts:
            for closer in ("]}", "}]}", "}", "]", "]}]}"):
                try:
                    return dedupe(json.loads(raw[:cut] + closer))
                except json.JSONDecodeError:
                    continue
        return None

    @staticmethod
    def _parse(raw: str, schema):
        """Pull the first JSON value out of the reply; small models often return a
        bare list where DeepEval's schema wants {"<field>": [...]}."""
        dec = json.JSONDecoder()
        for i, ch in enumerate(raw):
            if ch not in "[{":
                continue
            try:
                data, _ = dec.raw_decode(raw[i:])
            except json.JSONDecodeError:
                continue
            try:
                return schema(**data) if isinstance(data, dict) else schema(**{"": data})
            except Exception:
                fields = list(schema.model_fields)
                if len(fields) != 1:
                    continue
                items = data if isinstance(data, list) else [data]
                try:
                    return schema(**{fields[0]: items})
                except Exception:
                    continue
        repaired = OllamaJudge._repair(raw)
        if repaired is not None:
            try:
                return schema(**repaired) if isinstance(repaired, dict) else schema(**{list(schema.model_fields)[0]: repaired})
            except Exception:
                pass
        raise ValueError(f"no valid JSON for {schema.__name__}: {raw[:200]!r}")

    def generate(self, prompt: str, schema=None):
        if schema is None:
            return self._call(prompt, as_json=False)
        # Free-form first: forcing Ollama's `format` makes small models loop on
        # list outputs until num_predict is hit. Fall back to schema-constrained.
        try:
            return self._parse(self._call(prompt, as_json=False), schema)
        except Exception:
            return self._parse(self._call(prompt, as_json=True, schema=schema), schema)

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"ollama:{self.model_name}"

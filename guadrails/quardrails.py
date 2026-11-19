"""
Lightweight, dependency-free guardrails for the RAG pipeline.

Three checkpoints, each a pure function so they are easy to test and reuse:

    check_input(query)        -> before retrieval: length limits, prompt-injection
                                 / jailbreak detection, PII redaction in the query
    filter_context(chunks)    -> after retrieval: drop chunks carrying indirect
                                 prompt injection (instructions hidden in documents)
    check_output(answer, ...) -> after generation: PII redaction, system-prompt
                                 leak detection, citation (groundedness) check

Every check returns a GuardrailResult so the caller can see *why* something
was blocked or modified; nothing here calls the LLM, so it adds ~0 latency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ----------------------------------------------------------------- config
MAX_QUERY_CHARS = 2000
MIN_QUERY_CHARS = 3

# Direct prompt-injection / jailbreak phrasing in the user's question.
INJECTION_PATTERNS = [
    r"\bignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)\b",
    r"\bdisregard (all |any |the )?(previous|prior|above|system) (instructions|prompts?|rules)\b",
    r"\bforget (everything|all|your) (instructions|rules|above|you (were|have been) (told|given))\b",
    r"<\|\s*(im_start|im_end|system|endoftext)\s*\|>",
    r"\b(reveal|show|print|repeat|leak) (me )?(your |the )?(system|hidden|initial) (prompt|instructions|message)\b",
    r"\byou are now (a |an )?(?!assistant)\w+",
    r"\b(act|pretend) as (if you (are|were) )?(an? )?(unrestricted|unfiltered|jailbroken|dan)\b",
    r"\b(developer|god|dan|jailbreak) mode\b",
    r"\bdo anything now\b",
    r"\boverride (your |the )?(safety|guardrails?|rules|filters?)\b",
    r"<\s*/?\s*(system|assistant|im_start|im_end)\s*>",
    r"\[\s*(system|inst)\s*\]",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

# PII that should never be sent to the LLM or returned to the user.
PII_PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "CREDIT_CARD": r"\b[3-6]\d{3}(?:[ -]?\d{4}){2}[ -]?\d{1,4}\b",  # Visa/MC/Amex/Discover/RuPay prefixes
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "PHONE": r"(?<!\w)(?:\+?\d{1,3}[ -]?)?(?:\(?\d{3}\)?[ -]?)\d{3}[ -]?\d{4}(?!\w)",
    "API_KEY": r"\b(?:sk|gsk|hf|pk)[-_][A-Za-z0-9_\-]{16,}\b",
}
_PII_RE = {name: re.compile(p) for name, p in PII_PATTERNS.items()}

# Phrases that indicate the model is echoing its system prompt.
SYSTEM_LEAK_MARKERS = [
    "you are an accurate, helpful ai assistant",
    "[instructions]",
    "[context information]",
]

_CITATION_RE = re.compile(r"\[[^\[\]]+::[^\[\]]+::\d+\]|\[\d+\]")


# ---------------------------------------------------------------- results
@dataclass
class GuardrailResult:
    allowed: bool = True
    text: str = ""                       # possibly-redacted text to use downstream
    violations: list[str] = field(default_factory=list)  # hard failures (blocked)
    warnings: list[str] = field(default_factory=list)    # soft issues (allowed, flagged)

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "violations": self.violations, "warnings": self.warnings}


# ---------------------------------------------------------------- helpers
def detect_injection(text: str) -> list[str]:
    """Return the patterns that matched (empty list = clean)."""
    return [r.pattern for r in _INJECTION_RE if r.search(text)]


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)][::-1]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits))
    return len(digits) >= 13 and total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace PII with [REDACTED_<TYPE>] tags. Returns (clean_text, types_found)."""
    found = []
    for name, rx in _PII_RE.items():
        hit = False

        def _sub(m: re.Match, name=name) -> str:
            nonlocal hit
            if name == "CREDIT_CARD" and not _luhn_ok(m.group()):
                return m.group()  # e.g. a run of years/page numbers, not a card
            hit = True
            return f"[REDACTED_{name}]"

        text = rx.sub(_sub, text)
        if hit:
            found.append(name)
    return text, found


# -------------------------------------------------------- 1) input rails
def check_input(query: str) -> GuardrailResult:
    q = (query or "").strip()
    res = GuardrailResult(text=q)

    if len(q) < MIN_QUERY_CHARS:
        res.allowed = False
        res.violations.append(f"query_too_short (<{MIN_QUERY_CHARS} chars)")
        return res
    if len(q) > MAX_QUERY_CHARS:
        res.allowed = False
        res.violations.append(f"query_too_long (>{MAX_QUERY_CHARS} chars)")
        return res

    if detect_injection(q):
        res.allowed = False
        res.violations.append("prompt_injection_detected")
        return res

    res.text, pii = redact_pii(q)
    if pii:
        res.warnings.append(f"pii_redacted_from_query: {', '.join(pii)}")
    return res


# ------------------------------------------------------ 2) context rails
def filter_context(chunks: list[dict]) -> tuple[list[dict], GuardrailResult]:
    """
    Indirect prompt injection: a malicious PDF can contain text like
    "ignore previous instructions and ...". Such chunks are dropped before
    they reach the LLM prompt. (Chunk text is not PII-redacted, so numeric
    document content stays intact; PII is scrubbed from the answer instead.)
    """
    res = GuardrailResult()
    safe = []
    for c in chunks:
        if detect_injection(c.get("text", "") or ""):
            res.warnings.append(f"dropped_injected_chunk: {c.get('id', '?')}")
            continue
        safe.append(c)
    return safe, res


# ------------------------------------------------------- 3) output rails
def check_output(answer: str, chunks: list[dict] | None = None) -> GuardrailResult:
    res = GuardrailResult(text=answer or "")
    lowered = res.text.lower()

    if any(m in lowered for m in SYSTEM_LEAK_MARKERS):
        res.allowed = False
        res.violations.append("system_prompt_leak")
        res.text = "The generated answer was withheld by the output guardrail."
        return res

    res.text, pii = redact_pii(res.text)
    if pii:
        res.warnings.append(f"pii_redacted_from_answer: {', '.join(pii)}")

    if chunks and not _CITATION_RE.search(res.text):
        res.warnings.append("no_citations_in_answer")
    return res

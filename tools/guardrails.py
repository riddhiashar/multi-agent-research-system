"""Input and output guardrails (Day 28). Pure functions, no network, unit-tested.

check_input     -> is this question safe and sensible to send to paid APIs?
sanitize_output -> remove active content and secrets from LLM-written text before anyone sees or downloads it.

These are rule-based guard rails. They stop the common cases (obvious prompt-injection phrases, pasted API keys,
personal data, HTML/script, tracking images) but cannot catch every attack; they are one layer, next to the
password, the run cap and the output validator."""
import re
from typing import Dict, List

MIN_QUESTION_CHARS, MAX_QUESTION_CHARS = 10, 500
MAX_OUTPUT_CHARS = 60_000

_INVISIBLE = re.compile("[​-‏‪-‮⁠﻿]")     # zero-width and bidi control characters
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

_INJECTION = [re.compile(p, re.I) for p in (
    r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(instruction|prompt|rule|direction|guideline|guardrail)s?\b",
    r"\b(reveal|show|print|repeat|display|leak|output)\b.{0,40}\b(system prompt|your (instructions|prompt|rules)|hidden prompt|api[ _-]?keys?|secrets?)\b",
    r"\byou are now\b|\bact as (an? )?(unrestricted|jailbroken|dan)\b|\bdeveloper mode\b|\bjailbreak\b|\bdan mode\b",
    r"</?\s*(system|assistant|instructions?)\s*>|\[/?inst\]|^\s*system\s*:",
    r"\bnew instructions?\s*:",
)]
_HARMFUL = [re.compile(p, re.I) for p in (
    r"\b(make|build|synthesi[sz]e|produce|cook|manufacture)\b.{0,30}\b(bomb|explosives?|nerve agent|sarin|ricin|napalm|methamphetamine)\b",
    r"\b(write|create|build|code)\b.{0,30}\b(malware|ransomware|keylogger|botnet|ddos)\b",
    r"\bhow to\b.{0,15}\b(hack into|break into|steal)\b",
)]
_SECRETS = [re.compile(p) for p in (
    r"gsk_[A-Za-z0-9]{20,}", r"AIza[0-9A-Za-z_\-]{30,}", r"tvly-[A-Za-z0-9\-]{16,}", r"\bsk-[A-Za-z0-9_\-]{20,}",
    r"\bAKIA[0-9A-Z]{16}\b", r"\bghp_[A-Za-z0-9]{30,}", r"\bxox[bp]-[A-Za-z0-9\-]{10,}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----", r"(?i)\b(api[_-]?key|secret|password|token)\s*[:=]\s*\S{8,}",
)]
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_CAND = re.compile(r"\+?\d[\d\s().-]{8,}\d")
_CARD_CAND = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_AADHAAR = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")
_UNSAFE_SCHEME = re.compile(r"\b(javascript|vbscript|data|file):", re.I)
_URL = re.compile(r"https?://", re.I)


def luhn_ok(digits: str) -> bool:
    d = [int(c) for c in digits][::-1]
    return len(d) >= 13 and (sum(d[0::2]) + sum(sum(divmod(x * 2, 10)) for x in d[1::2])) % 10 == 0


def _has_phone(text: str) -> bool:
    for m in _PHONE_CAND.finditer(text):
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        if 10 <= len(digits) <= 15 and (raw.startswith("+") or re.search(r"[\s().-]", raw)) and not re.fullmatch(r"\d{4}[-\s]\d{4}", raw.strip()):
            return True
    return False


def _has_card(text: str) -> bool:
    return any(luhn_ok(re.sub(r"\D", "", m.group(0))) for m in _CARD_CAND.finditer(text))


def normalise(raw: str) -> str:
    return " ".join(_CONTROL.sub(" ", _INVISIBLE.sub("", raw or "")).split())


def check_input(raw: str) -> Dict:
    """-> {"ok", "clean", "error", "category", "flags"}. The error never repeats the offending text, so a pasted
    key is not echoed back to the screen."""
    clean = normalise(raw)
    flags: List[str] = []

    def block(category: str, message: str) -> Dict:
        return {"ok": False, "clean": clean, "error": message, "category": category, "flags": flags + [category]}

    if len(clean) < MIN_QUESTION_CHARS:
        return block("too_short", f"Please enter a research question of at least {MIN_QUESTION_CHARS} characters.")
    if len(clean) > MAX_QUESTION_CHARS:
        return block("too_long", f"Please keep the question under {MAX_QUESTION_CHARS} characters.")
    if _EMAIL.search(clean) or _SSN.search(clean) or _AADHAAR.search(clean) or _has_card(clean) or _has_phone(clean):
        return block("personal_data", "The question seems to contain personal data (email, phone, card or ID number). Remove it and try again.")
    if any(p.search(clean) for p in _SECRETS):
        return block("secret", "The question looks like it contains an API key or password. Remove it and try again.")
    if any(p.search(clean) for p in _INJECTION):
        return block("prompt_injection", "The question looks like an attempt to change the system's instructions. Ask a research question instead.")
    if any(p.search(clean) for p in _HARMFUL):
        return block("out_of_scope", "This tool researches publicly documented topics and can't help with that request.")
    if _UNSAFE_SCHEME.search(clean):
        return block("unsafe_link", "Links with javascript:, data: or file: schemes are not accepted.")
    letters = sum(c.isalpha() for c in clean)
    if letters / max(1, len(clean.replace(" ", ""))) < 0.6 or len(re.findall(r"[A-Za-z]{2,}", clean)) < 3:
        return block("not_a_question", "Please write the question as a sentence, for example: Compare battery recycling approaches of Tesla and CATL.")
    if _URL.search(clean):
        flags.append("contains_url")       # allowed: the question is text only and the URL is never fetched
    return {"ok": True, "clean": clean, "error": None, "category": None, "flags": flags}


# ----------------------------------------------------------------------------------------------- output
_BLOCK_TAGS = re.compile(r"<\s*(script|style|iframe|object|embed|form)\b.*?</\s*\1\s*>", re.I | re.S)
_TAG = re.compile(r"</?[A-Za-z][^<>\n]{0,200}>")
_PAREN = r"\((?:[^()]|\([^()]*\))*\)"          # (...) allowing one level of nested parentheses
_MD_IMAGE = re.compile(r"!\[[^\]]*\]" + _PAREN)
_UNSAFE_MD_LINK = re.compile(r"\[([^\]]*)\]\(\s*(?:javascript|vbscript|data|file):(?:[^()]|\([^()]*\))*\)", re.I)
_UNSAFE_BARE = re.compile(r"\b(?:javascript|vbscript):\S*", re.I)
_LEAK_PHRASES = re.compile(r"(system prompt|my instructions are|as an ai language model|i was instructed to)", re.I)


def sanitize_output(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> Dict:
    """-> {"text", "actions": [{"type","count"}]}. Removes active HTML, tracking images, unsafe links, secrets and
    e-mail addresses; flags (does not remove) phrases that suggest the model leaked its instructions."""
    actions: List[Dict] = []

    def note(kind: str, n: int) -> None:
        if n:
            actions.append({"type": kind, "count": n})

    t = _CONTROL.sub(lambda m: m.group(0) if m.group(0) in "\n\t" else " ", _INVISIBLE.sub("", text or ""))
    t, n = _BLOCK_TAGS.subn("", t); note("script_or_embed_removed", n)
    t, n = _MD_IMAGE.subn("", t); note("image_removed", n)
    t, n = _UNSAFE_MD_LINK.subn(r"\1", t); note("unsafe_link_removed", n)
    t, n = _UNSAFE_BARE.subn("[link removed]", t); note("unsafe_link_removed", n)
    t, n = _TAG.subn("", t); note("html_tag_removed", n)
    n = 0
    for p in _SECRETS:
        t, k = p.subn("[secret removed]", t); n += k
    note("secret_redacted", n)
    t, n = _EMAIL.subn("[email removed]", t); note("email_redacted", n)
    n = 0
    for m in list(_CARD_CAND.finditer(t)):
        if luhn_ok(re.sub(r"\D", "", m.group(0))):
            t = t.replace(m.group(0), "[number removed]"); n += 1
    note("card_number_redacted", n)
    note("possible_instruction_leak", len(_LEAK_PHRASES.findall(t)))
    if len(t) > max_chars:
        t = t[:max_chars].rsplit("\n", 1)[0] + "\n\n[report truncated: length limit]"
        note("truncated", 1)
    return {"text": t, "actions": actions}

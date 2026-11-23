#!/usr/bin/env python3
"""
Phonetic roster linker: resolve person-name mentions in ASR text to match roster entities.

This script implements the pipeline described below:

Step 0. Prepare two things
  - Qwen3: extract person-name-like mentions from EACH transcription segment.
  - Match roster: build a dictionary of entities (players/coaches/referees) for the match,
    including canonical names and common aliases (long/short names, last name, etc.).
  - Then for each mention, find the most similar roster entity.

Step 1. G2P (grapheme-to-phoneme): turn strings into phoneme tokens (or a phonetic code).
  - Backend A (recommended when available): phonemizer / g2p-en (optional dependencies).
  - Backend B (fallback, lightweight): Soundex (phonetic code, not true phonemes).
  - Backend C (not implemented): train/finetune custom G2P for heavy non-English names.

Step 2. Similarity on phoneme tokens (not characters)
  - sim_ph = 1 - edit_distance(phoneme1, phoneme2) / max(len1, len2)
  - In parallel, compute a char similarity sim_char (Jaro-Winkler).
  - Fuse: score = w_char * sim_char + w_ph * sim_ph

Step 3. Threshold decision
  - For each mention, compute top1/top2 entity scores.
  - Replace only if:
      top1 > tau_abs  AND  (top1 - top2) > tau_margin
    Otherwise output "UNCERTAIN".

Examples:
  # 1) Try one file (in-place edit)
  python /path/to/SoccerNet/asr/tools/phonetic_roster_linker.py \\
    --anno-file /path/to/SoccerNet/asr/annos/test/.../2-224p_processed_clean.json \\
    --text-field llm_polished_text \\
    --api-base http://localhost:8000/v1 \\
    --model Qwen3-235B-A22B-Instruct-2507

  # 2) Try ONE file under annos-root (in-place edit)
  python /path/to/SoccerNet/asr/tools/phonetic_roster_linker.py \\
    --annos-root /path/to/SoccerNet/asr/annos \\
    --try-one

  # 3) Process ALL files under annos-root (in-place edit)
  python /path/to/SoccerNet/asr/tools/phonetic_roster_linker.py \\
    --annos-root /path/to/SoccerNet/asr/annos

This script writes two new fields into each segment:
  - corrected_name_text: text with corrected names
  - name_replacements: list of {from,to,count} replacements used in that segment
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


def normalize_name(text: str) -> str:
    """
    Normalize for matching/dedup across languages.

    - Lowercase
    - Strip combining marks
    - Keep unicode alnum, collapse whitespace/punct to spaces
    """
    text = unicodedata.normalize("NFKD", text)
    out_chars: List[str] = []
    for ch in text:
        if unicodedata.combining(ch):
            continue
        if ch in {"’", "'"}:
            continue
        if ch.isalnum():
            out_chars.append(ch.lower())
            continue
        if ch in {"-", "_"}:
            out_chars.append(" ")
            continue
        if ch.isspace():
            out_chars.append(" ")
            continue
        out_chars.append(" ")
    norm = "".join(out_chars)
    norm = re.sub(r"\\s+", " ", norm).strip()
    return norm


def name_variants(name: str) -> Set[str]:
    base = normalize_name(name)
    if not base:
        return set()
    variants = {base}
    variants.add(base.replace(".", ""))
    tokens = base.split()
    if tokens:
        last = tokens[-1]
        last_name = tokens[-2] if (len(last) == 1 and len(tokens) >= 2) else last
        if len(last_name) > 1:
            variants.add(last_name)  # last-name-only (skip initials)
        if len(tokens) >= 2:
            variants.add(f"{tokens[0]} {last_name}")  # first + last
    return {v for v in variants if v}


@dataclass(frozen=True)
class RosterEntity:
    canonical: str
    role: str  # player/coach/referee/team
    team: Optional[str]
    aliases: Tuple[str, ...]

    @property
    def all_names(self) -> Tuple[str, ...]:
        return (self.canonical,) + self.aliases


def build_roster_entities(anno: Dict[str, Any], include_teams: bool = False) -> List[RosterEntity]:
    entities: Dict[str, RosterEntity] = {}

    def add_entity(canonical: str, role: str, team: Optional[str], aliases: Iterable[str]) -> None:
        canonical = canonical.strip()
        if not canonical:
            return
        all_aliases: Set[str] = set()
        for a in aliases:
            if a and a.strip() and a.strip() != canonical:
                all_aliases.add(a.strip())
        key = f"{role}:{team or ''}:{canonical}"
        if key in entities:
            merged = set(entities[key].aliases) | all_aliases
            entities[key] = RosterEntity(canonical=canonical, role=role, team=team, aliases=tuple(sorted(merged)))
        else:
            entities[key] = RosterEntity(canonical=canonical, role=role, team=team, aliases=tuple(sorted(all_aliases)))

    home_team = (anno.get("home") or {}).get("name") or anno.get("gameHomeTeam") or None
    away_team = (anno.get("away") or {}).get("name") or anno.get("gameAwayTeam") or None

    lineup = anno.get("lineup") or {}
    for side_key, team_name in (("home", home_team), ("away", away_team)):
        side = lineup.get(side_key) or {}
        for coach in (side.get("coach") or []):
            if not isinstance(coach, dict):
                continue
            canonical = (coach.get("name") or coach.get("long_name") or coach.get("short_name") or "").strip()
            raw_aliases = [coach.get("name"), coach.get("long_name"), coach.get("short_name")]
            derived: Set[str] = set()
            for a in raw_aliases:
                if a:
                    derived |= name_variants(a)
            add_entity(canonical=canonical, role="coach", team=team_name, aliases=sorted(set(filter(None, raw_aliases)) | derived))

        for player in (side.get("players") or []):
            if not isinstance(player, dict):
                continue
            canonical = (player.get("name") or player.get("long_name") or player.get("short_name") or "").strip()
            raw_aliases = [player.get("name"), player.get("long_name"), player.get("short_name")]
            derived = set()
            for a in raw_aliases:
                if a:
                    derived |= name_variants(a)
            add_entity(canonical=canonical, role="player", team=team_name, aliases=sorted(set(filter(None, raw_aliases)) | derived))

    for ref in (anno.get("referee") or []):
        if not isinstance(ref, str):
            continue
        add_entity(canonical=ref, role="referee", team=None, aliases=sorted(name_variants(ref)))

    if include_teams:
        for side_key in ("home", "away"):
            side = anno.get(side_key) or {}
            team_name = side.get("name") or ""
            if team_name:
                add_entity(canonical=team_name, role="team", team=team_name, aliases=sorted(side.get("names", []) or []))
        for team_name in (anno.get("teams") or []):
            add_entity(canonical=team_name, role="team", team=team_name, aliases=[])

    return list(entities.values())


def soundex_code(s: str) -> str:
    """
    Soundex (rough phonetic code). Good as a dependency-free fallback for "sounds like" recall.
    """
    s = normalize_name(s)
    s = re.sub(r"[^a-z]", "", s)
    if not s:
        return ""
    first = s[0].upper()
    mapping = {
        "b": "1",
        "f": "1",
        "p": "1",
        "v": "1",
        "c": "2",
        "g": "2",
        "j": "2",
        "k": "2",
        "q": "2",
        "s": "2",
        "x": "2",
        "z": "2",
        "d": "3",
        "t": "3",
        "l": "4",
        "m": "5",
        "n": "5",
        "r": "6",
    }
    digits: List[str] = []
    prev = ""
    for ch in s[1:]:
        code = mapping.get(ch, "0")
        if code != "0" and code != prev:
            digits.append(code)
        prev = code
    code = first + "".join(digits)
    code = (code + "000")[:4]
    return code


class G2P:
    def __init__(self, backend: str = "auto", language: str = "en-us"):
        self.backend = backend
        self.language = language
        self._init_backend()

    def _init_backend(self) -> None:
        self._phonemizer_backend = None
        self._phonemizer_separator = None
        self._g2p_en = None
        if self.backend in {"phonemizer", "auto"}:
            try:
                from phonemizer.backend import EspeakBackend  # type: ignore
                from phonemizer.separator import Separator  # type: ignore

                self._phonemizer_backend = EspeakBackend(
                    language=self.language, preserve_punctuation=False, with_stress=False
                )
                self._phonemizer_separator = Separator(phone=" ", word=" | ", syllable="")
                if self.backend == "auto":
                    self.backend = "phonemizer"
                    return
            except Exception:
                self._phonemizer_backend = None
                self._phonemizer_separator = None
        if self.backend in {"g2p_en", "auto"}:
            try:
                from g2p_en import G2p  # type: ignore

                self._g2p_en = G2p()
                if self.backend == "auto":
                    self.backend = "g2p_en"
                    return
            except Exception:
                self._g2p_en = None
        if self.backend == "auto":
            self.backend = "soundex"

    @lru_cache(maxsize=50_000)
    def phoneme_tokens(self, text: str) -> Tuple[str, ...]:
        text_norm = normalize_name(text)
        if not text_norm:
            return tuple()

        if self.backend == "phonemizer" and self._phonemizer_backend is not None and self._phonemizer_separator is not None:
            try:
                out = self._phonemizer_backend.phonemize([text_norm], separator=self._phonemizer_separator)[0]
                out = out.replace("|", " ")
                tokens = tuple(t for t in re.split(r"\\s+", out.strip()) if t)
                return tokens
            except Exception:
                pass

        if self.backend == "g2p_en" and self._g2p_en is not None:
            # g2p-en returns ARPAbet tokens; filter out spaces.
            try:
                tokens = [t for t in self._g2p_en(text_norm) if t and t != " "]
                return tuple(tokens)
            except Exception:
                pass

        if self.backend == "soundex":
            # Fallback: use last-name-ish token to avoid first-name noise (e.g., "Martin Atkinson" vs "Atkinson M.").
            tokens = [t for t in text_norm.split() if t]
            if not tokens:
                return tuple()
            last = tokens[-1]
            if len(last) == 1 and len(tokens) >= 2:
                last = tokens[-2]
            code = soundex_code(last)
            return (code,) if code else tuple()

        return tuple(t for t in text_norm.split() if t)


def levenshtein_distance(a: Sequence[str], b: Sequence[str]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def sim_edit(a: Sequence[str], b: Sequence[str]) -> float:
    denom = max(len(a), len(b))
    if denom == 0:
        return 1.0
    dist = levenshtein_distance(a, b)
    return 1.0 - (dist / denom)


def jaro_winkler(s1: str, s2: str, prefix_scale: float = 0.1) -> float:
    s1 = normalize_name(s1)
    s2 = normalize_name(s2)
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    match_distance = max(len1, len2) // 2 - 1

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    transpositions = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    transpositions /= 2

    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0

    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * prefix_scale * (1.0 - jaro)


@dataclass
class MatchResult:
    mention: str
    status: str  # MAPPED/UNCERTAIN
    canonical: Optional[str]
    role: Optional[str]
    team: Optional[str]
    score: float
    score_char: float
    score_ph: float
    top2_score: float


STOPWORDS_MINI: Set[str] = {
    # EN
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "in",
    "on",
    "at",
    "to",
    "from",
    "for",
    "of",
    "with",
    "as",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "i",
    "me",
    "my",
    "we",
    "us",
    "our",
    "you",
    "your",
    "he",
    "him",
    "his",
    "she",
    "her",
    "they",
    "them",
    "their",
    "not",
    "no",
    "yes",
    # ES (very common function words)
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "de",
    "del",
    "y",
    "o",
    "en",
    "por",
    "para",
    "con",
    "sin",
    "al",
    "se",
    "le",
    "les",
    "que",
    # FR/DE small set
    "et",
    "des",
    "du",
    "au",
    "aux",
    "der",
    "die",
    "das",
    "und",
    "im",
    "ein",
    "eine",
    "zu",
    "von",
    "mit",
    "auf",
    "fur",
    # fillers
    "oh",
    "uh",
    "um",
    "er",
}


def postfilter_mentions(
    names: Sequence[str],
    *,
    drop_stopwords: bool,
    min_len: int,
    min_single_token_len: int,
) -> List[str]:
    out: List[str] = []
    for raw in names:
        if not isinstance(raw, str):
            continue
        raw = raw.strip()
        if not raw:
            continue
        norm = normalize_name(raw)
        if not norm:
            continue
        if drop_stopwords and norm in STOPWORDS_MINI:
            continue
        # Apply length filters mainly for ASCII-ish outputs; non-ASCII names can be short (e.g., 2-char names).
        if norm.isascii() and len(norm) < min_len:
            continue
        tokens = norm.split()
        if len(tokens) == 1 and tokens[0].isascii() and len(tokens[0]) < min_single_token_len:
            continue
        out.append(raw)
    return out


def dedup_preserve_order(names: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for n in names:
        k = normalize_name(n)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(n)
    return out


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _extract_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    return text[start : end + 1]


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json)?\\s*(.*?)\\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def _repair_json(text: str) -> str:
    """
    Best-effort JSON repair for common LLM formatting bugs:
    - trailing commas
    - Python booleans/None
    - stray code fences
    """
    t = _strip_code_fences(text)
    t = _extract_json_object(t) or t
    t = t.replace("\ufeff", "")
    # Remove illegal control chars (keep \\t\\n\\r)
    t = re.sub(r"[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f]", "", t)
    # Fix trailing commas in objects/arrays: { "a": 1, } / [1,2,]
    t = re.sub(r",\\s*([}\\]])", r"\\1", t)
    # Python literals -> JSON literals
    t = re.sub(r"\\bNone\\b", "null", t)
    t = re.sub(r"\\bTrue\\b", "true", t)
    t = re.sub(r"\\bFalse\\b", "false", t)
    return t


def parse_llm_results(raw: str) -> Dict[int, List[str]]:
    """
    Parse LLM response into: {segment_id: [names...]}.

    Raises on unrecoverable parse errors.
    """
    candidates = []
    candidates.append(raw)
    candidates.append(_strip_code_fences(raw))
    candidates.append(_extract_json_object(raw) or raw)
    candidates.append(_extract_json_object(_strip_code_fences(raw)) or _strip_code_fences(raw))
    last_err: Optional[Exception] = None
    for cand in candidates:
        try:
            data = json.loads(cand)
            break
        except Exception as e:
            last_err = e
            continue
    else:
        repaired = _repair_json(raw)
        try:
            data = json.loads(repaired)
        except Exception as e:
            raise ValueError(f"LLM output is not valid JSON (even after repair): {e}") from last_err

    results = data.get("results") or []
    out: Dict[int, List[str]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        sid = item.get("segment_id")
        names = item.get("names")
        if isinstance(sid, int) and isinstance(names, list):
            out[sid] = [n for n in names if isinstance(n, str) and n.strip()]
    return out


QWEN_PROMPT = """You are extracting PERSON NAMES from soccer match commentary ASR text.

Rules:
- Only extract person names (players/coaches/referees/commentators) that appear in the text.
- DO NOT extract teams, stadiums, leagues, countries, competitions.
- Keep each extracted name as an exact substring from the input text (keep casing/spaces).
- If there are no person names, output an empty list for that segment.
- Output MUST be valid JSON only (no markdown, no explanations).

Input segments JSON:
{segments_json}

Output JSON schema:
{{
  "results": [
    {{"segment_id": <int>, "names": ["<name1>", "<name2>", ...]}},
    ...
  ]
}}
"""


@dataclass
class LLMConfig:
    api_base: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float
    max_retries: int
    retry_delay: float
    request_delay: float


def create_client(config: LLMConfig) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package not found. Install with: pip install openai")
    return OpenAI(base_url=config.api_base, api_key="dummy", timeout=config.timeout)


def call_llm(client: Any, config: LLMConfig, prompt: str) -> Optional[str]:
    for attempt in range(config.max_retries):
        try:
            resp = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            content = resp.choices[0].message.content if resp.choices else None
            if content:
                if config.request_delay > 0:
                    time.sleep(config.request_delay)
                return content.strip()
            return None
        except Exception as e:
            if attempt < config.max_retries - 1:
                print(f"    LLM API error (attempt {attempt+1}/{config.max_retries}): {e}")
                time.sleep(config.retry_delay)
            else:
                print(f"    LLM API error (all retries failed): {e}")
                return None
    return None


def best_match_for_mention(
    mention: str,
    roster: Sequence[RosterEntity],
    g2p: G2P,
    w_char: float = 0.4,
    w_ph: float = 0.6,
) -> Tuple[Optional[RosterEntity], float, float, float, float]:
    mention_norm = normalize_name(mention)
    mention_ph = g2p.phoneme_tokens(mention_norm)

    best_entity = None
    best_score = -1.0
    best_char = 0.0
    best_ph = 0.0
    second_best = -1.0

    for ent in roster:
        ent_best = -1.0
        ent_best_char = 0.0
        ent_best_ph = 0.0
        for cand in ent.all_names:
            cand_norm = normalize_name(cand)
            if not cand_norm:
                continue
            score_char = jaro_winkler(mention_norm, cand_norm)
            cand_ph = g2p.phoneme_tokens(cand_norm)
            if mention_ph and cand_ph:
                score_ph = sim_edit(mention_ph, cand_ph)
                score = w_char * score_char + w_ph * score_ph
            else:
                # If phoneme extraction fails (common for non-Latin names under soundex),
                # fall back to pure character similarity instead of treating empty tokens as a perfect match.
                score_ph = 0.0
                score = score_char
            if score > ent_best:
                ent_best = score
                ent_best_char = score_char
                ent_best_ph = score_ph

        if ent_best > best_score:
            second_best = best_score
            best_score = ent_best
            best_entity = ent
            best_char = ent_best_char
            best_ph = ent_best_ph
        elif ent_best > second_best:
            second_best = ent_best

    return best_entity, best_score, best_char, best_ph, max(0.0, second_best)


def resolve_mentions(
    mentions: Sequence[str],
    roster: Sequence[RosterEntity],
    g2p: G2P,
    tau_abs: float,
    tau_margin: float,
    w_char: float = 0.4,
    w_ph: float = 0.6,
) -> List[MatchResult]:
    results: List[MatchResult] = []
    for mention in mentions:
        ent, score, score_char, score_ph, score2 = best_match_for_mention(
            mention=mention,
            roster=roster,
            g2p=g2p,
            w_char=w_char,
            w_ph=w_ph,
        )
        if ent is not None and (score >= tau_abs) and ((score - score2) >= tau_margin):
            results.append(
                MatchResult(
                    mention=mention,
                    status="MAPPED",
                    canonical=ent.canonical,
                    role=ent.role,
                    team=ent.team,
                    score=score,
                    score_char=score_char,
                    score_ph=score_ph,
                    top2_score=score2,
                )
            )
        else:
            results.append(
                MatchResult(
                    mention=mention,
                    status="UNCERTAIN",
                    canonical=ent.canonical if ent is not None else None,
                    role=ent.role if ent is not None else None,
                    team=ent.team if ent is not None else None,
                    score=score if score >= 0 else 0.0,
                    score_char=score_char,
                    score_ph=score_ph,
                    top2_score=score2,
                )
            )
    return results


def _iter_segments(
    anno: Dict[str, Any],
    *,
    text_field: str,
    limit_segments: Optional[int],
    max_chars: Optional[int],
) -> Iterable[Tuple[int, Dict[str, Any], str]]:
    segs = anno.get("segments") or []
    for i, seg in enumerate(segs):
        if limit_segments is not None and i >= limit_segments:
            break
        if not isinstance(seg, dict):
            continue
        seg_id = seg.get("segment_id")
        if not isinstance(seg_id, int):
            seg_id = i
        val = seg.get(text_field)
        if not isinstance(val, str) or not val.strip():
            continue
        text = val.strip()
        if max_chars is not None and max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars]
        yield seg_id, seg, text


def _extract_mentions_qwen3(
    *,
    anno: Dict[str, Any],
    text_field: str,
    limit_segments: Optional[int],
    max_chars: int,
    client: Any,
    llm_config: LLMConfig,
    llm_batch_size: int,
    cache: Optional[Dict[str, List[str]]],
    drop_stopwords: bool,
    min_mention_len: int,
    min_single_token_len: int,
) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    failed_seg_ids: Set[int] = set()
    pending: List[Tuple[int, str, str]] = []

    def _process_batch(batch: List[Tuple[int, str, str]]) -> Tuple[Dict[int, List[str]], bool]:
        segments_payload = [{"segment_id": sid, "text": txt} for sid, _, txt in batch]
        prompt = QWEN_PROMPT.format(segments_json=json.dumps(segments_payload, ensure_ascii=False))
        raw = call_llm(client, llm_config, prompt)
        if not raw:
            return {}, False
        try:
            return parse_llm_results(raw), True
        except Exception as e:
            if len(batch) > 1:
                mid = len(batch) // 2
                left, ok_left = _process_batch(batch[:mid])
                right, ok_right = _process_batch(batch[mid:])
                merged = dict(left)
                merged.update(right)
                return merged, (ok_left or ok_right)
            print(f"    LLM parse error for segment_id={batch[0][0]}: {e}")
            return {}, False

    def flush_pending() -> None:
        nonlocal pending, cache, out, failed_seg_ids
        if not pending:
            return

        parsed, ok = _process_batch(pending)
        if not ok:
            for sid, _, _ in pending:
                failed_seg_ids.add(sid)
            pending = []
            return
        for sid, key, txt in pending:
            names = parsed.get(sid, [])
            names = postfilter_mentions(
                names,
                drop_stopwords=drop_stopwords,
                min_len=min_mention_len,
                min_single_token_len=min_single_token_len,
            )
            names = [n for n in names if n in txt]
            names = dedup_preserve_order(names)
            out[sid] = names
            if cache is not None:
                cache[key] = names

        pending = []

    for seg_id, _, seg_text in _iter_segments(anno, text_field=text_field, limit_segments=limit_segments, max_chars=max_chars):
        key = sha1_text(seg_text)
        if cache is not None and key in cache:
            out[seg_id] = dedup_preserve_order(cache[key])
            continue
        pending.append((seg_id, key, seg_text))
        if len(pending) >= llm_batch_size:
            flush_pending()
    flush_pending()

    if failed_seg_ids:
        raise RuntimeError(f"LLM extraction failed for {len(failed_seg_ids)} segments")
    return out


def _is_word_char(ch: str) -> bool:
    return bool(ch) and (ch.isalnum() or ch == "_")


def _apply_replacements(text: str, replacements: Sequence[Tuple[str, str]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Replace mentions with canonical names in a single segment, avoiding overlaps.

    Returns: (new_text, applied_mappings)
    """
    if not text or not replacements:
        return text, []

    spans: List[Tuple[int, int, str, str]] = []
    for src, dst in replacements:
        if not src or src == dst:
            continue
        for m in re.finditer(re.escape(src), text):
            s, e = m.start(), m.end()
            before = text[s - 1] if s > 0 else ""
            after = text[e] if e < len(text) else ""
            if _is_word_char(before) and _is_word_char(src[0]):
                continue
            if _is_word_char(after) and _is_word_char(src[-1]):
                continue
            spans.append((s, e, src, dst))

    if not spans:
        return text, []

    spans.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    chosen: List[Tuple[int, int, str, str]] = []

    def overlaps(a: Tuple[int, int, str, str], b: Tuple[int, int, str, str]) -> bool:
        return not (a[1] <= b[0] or b[1] <= a[0])

    for span in spans:
        if any(overlaps(span, kept) for kept in chosen):
            continue
        chosen.append(span)

    chosen.sort(key=lambda x: x[0], reverse=True)
    out_text = text
    chosen_forward = sorted(chosen, key=lambda x: x[0])
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _, _, src, dst in chosen_forward:
        k = (src, dst)
        if k not in merged:
            merged[k] = {"from": src, "to": dst, "count": 1}
        else:
            merged[k]["count"] = int(merged[k]["count"]) + 1

    for s, e, src, dst in chosen:
        repl = dst if (dst.startswith("[") and dst.endswith("]")) else f"[{dst}]"
        out_text = out_text[:s] + repl + out_text[e:]
    return out_text, list(merged.values())


def _load_cache(cache_path: Path) -> Dict[str, List[str]]:
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            out: Dict[str, List[str]] = {}
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, list) and all(isinstance(x, str) for x in v):
                    out[k] = v
            return out
    except Exception:
        pass
    return {}


def _save_cache(cache_path: Path, cache: Dict[str, List[str]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _iter_anno_files(annos_root: Path) -> List[Path]:
    return sorted(p for p in annos_root.rglob("*.json") if p.is_file())


def _process_anno_file(
    anno_path: Path,
    *,
    text_field: str,
    limit_segments: Optional[int],
    max_chars: int,
    include_teams: bool,
    g2p: G2P,
    llm_client: Any,
    llm_config: LLMConfig,
    llm_batch_size: int,
    cache: Optional[Dict[str, List[str]]],
    drop_stopwords: bool,
    min_mention_len: int,
    min_single_token_len: int,
    tau_abs: float,
    tau_margin: float,
    w_char: float,
    w_ph: float,
) -> Dict[str, Any]:
    try:
        anno = json.loads(anno_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"path": str(anno_path), "skipped": True, "reason": f"json_load_error: {e}"}

    if not isinstance(anno, dict) or not isinstance(anno.get("segments"), list):
        return {"path": str(anno_path), "skipped": True, "reason": "not_a_match_anno"}

    roster = build_roster_entities(anno, include_teams=include_teams)

    try:
        mentions_by_seg = _extract_mentions_qwen3(
            anno=anno,
            text_field=text_field,
            limit_segments=limit_segments,
            max_chars=max_chars,
            client=llm_client,
            llm_config=llm_config,
            llm_batch_size=llm_batch_size,
            cache=cache,
            drop_stopwords=drop_stopwords,
            min_mention_len=min_mention_len,
            min_single_token_len=min_single_token_len,
        )
    except Exception as e:
        return {"path": str(anno_path), "skipped": True, "reason": f"llm_failed: {e}"}

    total_segments = 0
    segments_with_mentions = 0
    segments_with_replacements = 0
    total_replacements = 0

    segs = anno.get("segments") or []
    for i, seg in enumerate(segs):
        if limit_segments is not None and i >= limit_segments:
            break
        if not isinstance(seg, dict):
            continue
        total_segments += 1

        seg_id = seg.get("segment_id")
        if not isinstance(seg_id, int):
            seg_id = i

        raw_text = seg.get(text_field)
        seg_text = raw_text if isinstance(raw_text, str) else ""

        mentions = mentions_by_seg.get(seg_id, [])
        if mentions:
            segments_with_mentions += 1

        replacements: List[Tuple[str, str]] = []
        if roster and mentions and seg_text:
            resolved = resolve_mentions(
                mentions=mentions,
                roster=roster,
                g2p=g2p,
                tau_abs=tau_abs,
                tau_margin=tau_margin,
                w_char=w_char,
                w_ph=w_ph,
            )
            for r in resolved:
                if r.status != "MAPPED" or not r.canonical:
                    continue
                if normalize_name(r.mention) == normalize_name(r.canonical):
                    continue
                if r.mention not in seg_text:
                    continue
                replacements.append((r.mention, r.canonical))

        corrected, applied = _apply_replacements(seg_text, replacements)
        seg["corrected_name_text"] = corrected
        seg["name_replacements"] = applied

        if applied:
            segments_with_replacements += 1
            total_replacements += len(applied)

    _write_json_atomic(anno_path, anno)
    return {
        "path": str(anno_path),
        "skipped": False,
        "segments": total_segments,
        "with_mentions": segments_with_mentions,
        "with_replacements": segments_with_replacements,
        "total_replacements": total_replacements,
    }


def main() -> None:
    anno_file = "/path/to/SoccerNet/asr/annos/train/england_epl_2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1-224p_processed_clean.json"
    parser = argparse.ArgumentParser(
        description="Extract person mentions with Qwen3, link to roster by phonetic similarity, and write corrected fields in-place."
    )
    parser.add_argument("--anno-file", type=Path, default=None, help="Process ONE processed match annotation JSON (in-place).")
    parser.add_argument(
        "--annos-root",
        type=Path,
        default="/path/to/SoccerNet/asr/annos",
        help="Process ALL *.json files under this directory recursively (in-place).",
    )
    parser.add_argument("--try-one", action="store_true", help="When using --annos-root: only process the first file (for a quick try).")
    parser.add_argument(
        "--text-field",
        type=str,
        default="llm_polished_text",
        help="Which segment text field to use (e.g., llm_polished_text/original_text/corrected_text).",
    )
    parser.add_argument("--limit-segments", type=int, default=None, help="Only use first N segments for quick checks.")
    parser.add_argument("--max-chars", type=int, default=400, help="Truncate each segment text to reduce context/latency.")
    parser.add_argument("--include-teams", action="store_true", help="Also include team names in the roster.")

    parser.add_argument("--g2p-backend", type=str, default="auto", choices=["auto", "phonemizer", "g2p_en", "soundex"], help="G2P backend.")
    parser.add_argument("--g2p-language", type=str, default="en-us", help="phonemizer language (e.g., en-us, fr-fr).")
    parser.add_argument("--w-char", type=float, default=0.01, help="Weight for character similarity.")
    parser.add_argument("--w-ph", type=float, default=0.99, help="Weight for phoneme similarity.")

    parser.add_argument("--tau-abs", type=float, default=0.3, help="Absolute threshold for mapping.")
    parser.add_argument("--tau-margin", type=float, default=0.03, help="Margin threshold between top1 and top2.")

    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1", help="OpenAI-compatible API base.")
    parser.add_argument("--model", type=str, default="Qwen3-235B-A22B-Instruct-2507", help="LLM model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    parser.add_argument("--max-tokens", type=int, default=1024, help="LLM max tokens.")
    parser.add_argument("--timeout", type=float, default=10.0, help="LLM request timeout seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="LLM max retries.")
    parser.add_argument("--retry-delay", type=float, default=1.0, help="Sleep seconds between retries.")
    parser.add_argument("--request-delay", type=float, default=0.0, help="Optional sleep after each successful request.")
    parser.add_argument("--llm-batch-size", type=int, default=25, help="Segments per LLM request.")
    parser.add_argument("--cache", type=Path, default=None, help="Optional JSON cache for qwen3 extraction (keyed by text sha1).")

    parser.add_argument(
        "--drop-stopwords",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Post-filter: drop common function words (recommended).",
    )
    parser.add_argument("--min-mention-len", type=int, default=3, help="Post-filter: drop mentions shorter than this (after normalize).")
    parser.add_argument(
        "--min-single-token-len",
        type=int,
        default=4,
        help="Post-filter: if a mention is a single token, require at least this length.",
    )
    args = parser.parse_args()

    if args.anno_file is None and args.annos_root is None:
        raise SystemExit("Provide --anno-file or --annos-root.")
    if args.anno_file is not None and args.annos_root is not None:
        raise SystemExit("Use only one of --anno-file or --annos-root.")

    g2p = G2P(backend=args.g2p_backend, language=args.g2p_language)
    llm_config = LLMConfig(
        api_base=args.api_base,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        request_delay=args.request_delay,
    )
    llm_client = create_client(llm_config)

    cache: Optional[Dict[str, List[str]]] = None
    if args.cache is not None:
        cache = _load_cache(args.cache) if args.cache.exists() else {}

    if args.anno_file is not None:
        paths = [args.anno_file]
    else:
        paths = _iter_anno_files(args.annos_root)
        if args.try_one:
            paths = paths[:1]

    if not paths:
        raise SystemExit("No anno files found.")

    processed = 0
    skipped = 0
    for i, p in enumerate(paths, start=1):
        stats = _process_anno_file(
            p,
            text_field=args.text_field,
            limit_segments=args.limit_segments,
            max_chars=args.max_chars,
            include_teams=args.include_teams,
            g2p=g2p,
            llm_client=llm_client,
            llm_config=llm_config,
            llm_batch_size=args.llm_batch_size,
            cache=cache,
            drop_stopwords=args.drop_stopwords,
            min_mention_len=args.min_mention_len,
            min_single_token_len=args.min_single_token_len,
            tau_abs=args.tau_abs,
            tau_margin=args.tau_margin,
            w_char=args.w_char,
            w_ph=args.w_ph,
        )
        if stats.get("skipped"):
            skipped += 1
            print(f"[{i}/{len(paths)}] SKIP {p} ({stats.get('reason')})")
            continue
        processed += 1
        print(
            f"[{i}/{len(paths)}] OK {p} "
            f"segments={stats['segments']} mentions={stats['with_mentions']} "
            f"repl_segments={stats['with_replacements']} repl_pairs={stats['total_replacements']}"
        )

        if args.cache is not None and cache is not None:
            _save_cache(args.cache, cache)

    print(f"done: processed={processed}, skipped={skipped}, total={len(paths)}")


if __name__ == "__main__":
    main()

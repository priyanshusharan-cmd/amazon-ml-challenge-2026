"""Bounded-batch direct evidence that preserves information lost by normalization.

``add_features(df)`` accepts q_name, q_addr, s_name, s_addr (raw strings),
q_core, s_core (normalized names), and q_anorm, s_anorm (normalized addresses).
It returns q_idx/s1_idx when present, followed by FEATURES as Float32 columns.
No labels, catalog statistics, country, or other query records are consulted.

Call on batches of roughly 50,000--200,000 pairs. Cached text parsing is bounded;
RapidFuzz comparisons use one worker so callers control CPU/resource concurrency.
Number roles are deliberately conservative: a premise must be leading or explicitly
labelled. A later unlabeled number may be a postcode, so it is not called a premise.
Role labels cover common English/French terms already supported by the project.
Unrecognized formats remain unknown, not an inferred match or conflict.
"""
from __future__ import annotations

from functools import lru_cache
import re
from typing import NamedTuple
import unicodedata

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

INPUTS = ["q_name", "q_addr", "s_name", "s_addr", "q_core", "s_core", "q_anorm", "s_anorm"]
FEATURES = [
    "d3_prem_known", "d3_prem_one_known", "d3_prem_equal", "d3_prem_conflict",
    "d3_prem_sim", "d3_prem_numset_jaccard", "d3_prem_suffix_conflict",
    "d3_unit_known", "d3_unit_equal", "d3_unit_conflict", "d3_unit_one_known",
    "d3_floor_known", "d3_floor_equal", "d3_floor_conflict", "d3_role_cross_match",
    "d3_addr_alpha_ratio", "d3_addr_alpha_token_set", "d3_addr_alpha_diff_ratio",
    "d3_addr_qonly_fuzzy", "d3_addr_sonly_fuzzy",
    "d3_name_qonly_fuzzy", "d3_name_sonly_fuzzy", "d3_name_acronym_equal",
    "d3_name_raw_key_ratio", "d3_addr_raw_key_ratio",
]

_SPECIAL = str.maketrans({"ß": "ss", "œ": "oe", "æ": "ae", "ø": "o", "ł": "l", "đ": "d"})
_SPACE = re.compile(r"\s+")
_KEY = re.compile(r"[^a-z0-9]+")
_ALPHA = re.compile(r"[a-z]+")
_NUM = re.compile(r"\d+")
# A numeric identifier can preserve compound house identifiers (1-4-283/3),
# suffix letters (12b), and bis/ter. Ordinals are recognized only as floors.
_IDENT = re.compile(
    r"(?<![a-z0-9])\d+(?:(?:st|nd|rd|th|eme|er)\b|[a-z]\b)?"
    r"(?:\s*[-/]\s*\d+[a-z]?\b)*(?:\s+(?:bis|ter|quater)\b)?(?![a-z0-9])"
)
_ORDINAL = re.compile(r"^\d+(?:st|nd|rd|th|eme|er|e)$")
_UNIT_WORD = r"(?:apartment|appartement|appt|apt|suite|ste|unit|unite|flat|room|rm|local|lot|batiment)"
_FLOOR_WORD = r"(?:floor|flr|fl|etage|niveau)"
_UNIT_BEFORE = re.compile(r"\b" + _UNIT_WORD + r"\s*(?:(?:no|number|n)\s*)?[#:]?\s*$")
_UNIT_AFTER = re.compile(r"^\s*" + _UNIT_WORD + r"\b")
_FLOOR_BEFORE = re.compile(r"\b" + _FLOOR_WORD + r"\s*(?:(?:no|number|n)\s*)?[#:]?\s*$")
_FLOOR_AFTER = re.compile(r"^\s*" + _FLOOR_WORD + r"\b")
_HOUSE_BEFORE = re.compile(
    r"\b(?:(?:house|h|door|d|plot|premise|premises)\s*(?:no|number|n)?|no|number|n)\s*[#:]?\s*$"
)
_LETTER_UNIT = re.compile(r"\b" + _UNIT_WORD + r"\s*[#:]?\s*([a-z]{1,3})(?![a-z0-9])")
_FOLLOWING_IDENTIFIER = re.compile(r"^\s*(?:(?:no|number|n)\s*)?[#:]?\s*(?:\d|[a-z]{1,3}\b)")


class AddressParts(NamedTuple):
    premise: str
    units: tuple[str, ...]
    floors: tuple[str, ...]


@lru_cache(maxsize=32768)
def _fold(raw: str) -> str:
    text = unicodedata.normalize("NFKD", raw.lower().translate(_SPECIAL))
    out = []
    for ch in text:
        if ord(ch) < 128:
            out.append(ch)
            continue
        if unicodedata.combining(ch):
            continue
        try:
            out.append(str(unicodedata.decimal(ch)))
        except (ValueError, TypeError):
            out.append(ch)
    return _SPACE.sub(" ", "".join(out)).strip()


def _canonical_number(token: str, floor: bool = False) -> str:
    token = re.sub(r"\s+", "", token)
    if floor:
        token = re.sub(r"(?<=\d)(?:st|nd|rd|th|eme|er|e)$", "", token)
    return _NUM.sub(lambda m: str(int(m.group())), token)


def _role_after(after: str, pattern: re.Pattern) -> bool:
    match = pattern.match(after)
    # In '12 apt 4', apt labels the following 4, not the preceding house 12.
    return bool(match and not _FOLLOWING_IDENTIFIER.match(after[match.end():]))


@lru_cache(maxsize=32768)
def address_parts(raw: str) -> AddressParts:
    """Parse explicit roles without discarding their labels or number suffixes."""
    text = _fold(raw).replace(".", " ").replace("°", " ").replace("º", " ")
    premise = ""
    units: set[str] = set()
    floors: set[str] = set()
    for match in _IDENT.finditer(text):
        before, after = text[:match.start()], text[match.end():]
        token = match.group()
        if _FLOOR_BEFORE.search(before) or _role_after(after, _FLOOR_AFTER):
            floors.add(_canonical_number(token, floor=True))
        elif _UNIT_BEFORE.search(before) or _role_after(after, _UNIT_AFTER):
            units.add(_canonical_number(token))
        elif "#" in before[-3:] and re.search(r"[a-z0-9]", before[:-3]):
            units.add(_canonical_number(token))
        elif not _ORDINAL.fullmatch(token) and not premise:
            leading = not re.search(r"[a-z0-9]", before)
            if leading or _HOUSE_BEFORE.search(before):
                premise = _canonical_number(token)
    for match in _LETTER_UNIT.finditer(text):
        # 'unit no 5' is a numbered unit, not the alphabetic identifier 'no'.
        if match.group(1) not in {"no", "n", "the"}:
            units.add(match.group(1))
    return AddressParts(premise, tuple(sorted(units)), tuple(sorted(floors)))


def _role_values(q, s) -> tuple[float, float, float, float]:
    both = bool(q) and bool(s)
    equal = both and q == s
    return float(both), float(equal), float(both and not equal), float(bool(q) != bool(s))


def _fuzzy_unmatched(q: str, s: str) -> tuple[float, float, str, str]:
    """Symmetric mean best token match for tokens absent from the other side.

    Empty differences score 1 only when the corresponding source has tokens.
    Empty strings carry no positive evidence. Short tokens use the same literal
    edit metric, so an initial is not automatically expanded into a full word.
    """
    qt, st = set(q.split()), set(s.split())
    qo, so = sorted(qt - st), sorted(st - qt)
    def coverage(left: list[str], right: set[str], original: set[str]) -> float:
        if not original or not right:
            return 0.0
        if not left:
            return 1.0
        return sum(max(fuzz.ratio(t, u) for u in right) for t in left) / (100.0 * len(left))
    return coverage(qo, st, qt), coverage(so, qt, st), " ".join(qo), " ".join(so)


def _cp(a: list[str], b: list[str], scorer=fuzz.ratio) -> np.ndarray:
    values = process.cpdist(a, b, scorer=scorer, workers=1, dtype=np.float32) / 100.0
    # RapidFuzz ratio('', '') == 100, which would turn absence into agreement.
    if len(values):
        values[np.fromiter((not x or not y for x, y in zip(a, b)), bool, count=len(a))] = 0.0
    return values


def _acronym(name: str) -> str:
    parts = name.split()
    return "".join(t[0] for t in parts) if len(parts) >= 2 else ""


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    missing = [col for col in INPUTS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing direct feature input columns: {missing}")
    ids = [c for c in ("q_idx", "s1_idx") if c in df.columns]
    if not df.height:
        return df.select(ids).with_columns([pl.Series(c, [], dtype=pl.Float32) for c in FEATURES])
    strings = {col: df[col].fill_null("").cast(pl.String).to_list() for col in INPUTS}
    n = df.height
    out = {col: np.empty(n, dtype=np.float32) for col in FEATURES}
    qa = [" ".join(_ALPHA.findall(s)) for s in strings["q_anorm"]]
    sa = [" ".join(_ALPHA.findall(s)) for s in strings["s_anorm"]]
    qa_diff, sa_diff = [], []
    for i in range(n):
        qp, sp = address_parts(strings["q_addr"][i]), address_parts(strings["s_addr"][i])
        pk, pe, pc, po = _role_values(qp.premise, sp.premise)
        for key, val in zip(("known", "equal", "conflict", "one_known"), (pk, pe, pc, po)):
            out[f"d3_prem_{key}"][i] = val
        out["d3_prem_sim"][i] = fuzz.ratio(qp.premise, sp.premise) / 100.0 if pk else 0.0
        qnums, snums = set(_NUM.findall(qp.premise)), set(_NUM.findall(sp.premise))
        out["d3_prem_numset_jaccard"][i] = len(qnums & snums) / max(1, len(qnums | snums))
        out["d3_prem_suffix_conflict"][i] = float(pk and qnums == snums and qp.premise != sp.premise)
        uk, ue, uc, uo = _role_values(qp.units, sp.units)
        for key, val in zip(("known", "equal", "conflict", "one_known"), (uk, ue, uc, uo)):
            out[f"d3_unit_{key}"][i] = val
        fk, fe, fc, _ = _role_values(qp.floors, sp.floors)
        for key, val in zip(("known", "equal", "conflict"), (fk, fe, fc)):
            out[f"d3_floor_{key}"][i] = val
        out["d3_role_cross_match"][i] = float(
            bool(qp.premise) and qp.premise in sp.units or bool(sp.premise) and sp.premise in qp.units
        )
        qc, sc, qd, sd = _fuzzy_unmatched(qa[i], sa[i])
        out["d3_addr_qonly_fuzzy"][i], out["d3_addr_sonly_fuzzy"][i] = qc, sc
        qa_diff.append(qd); sa_diff.append(sd)
        qc, sc, _, _ = _fuzzy_unmatched(strings["q_core"][i], strings["s_core"][i])
        out["d3_name_qonly_fuzzy"][i], out["d3_name_sonly_fuzzy"][i] = qc, sc
        qcore, score = strings["q_core"][i], strings["s_core"][i]
        qacr, sacr = _acronym(qcore), _acronym(score)
        out["d3_name_acronym_equal"][i] = float(
            bool(qacr) and qacr == score.replace(" ", "") or bool(sacr) and sacr == qcore.replace(" ", "")
        )
    out["d3_addr_alpha_ratio"] = _cp(qa, sa)
    out["d3_addr_alpha_token_set"] = _cp(qa, sa, fuzz.token_set_ratio)
    out["d3_addr_alpha_diff_ratio"] = _cp(qa_diff, sa_diff)
    for i, (q, s) in enumerate(zip(qa_diff, sa_diff)):
        if not q and not s and qa[i] and sa[i]:
            out["d3_addr_alpha_diff_ratio"][i] = 1.0
    for source, target in (("name", "d3_name_raw_key_ratio"), ("addr", "d3_addr_raw_key_ratio")):
        q = [_KEY.sub("", _fold(v)) for v in strings[f"q_{source}"]]
        s = [_KEY.sub("", _fold(v)) for v in strings[f"s_{source}"]]
        out[target] = _cp(q, s)
    return df.select(ids).with_columns([pl.Series(c, out[c], dtype=pl.Float32) for c in FEATURES])

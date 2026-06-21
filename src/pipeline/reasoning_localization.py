"""Polish public reasoning text without touching machine-scored fields.

Structured player fields should keep provider/API names so graders can match
them against truth data. The public `reasoning.*` strings, however, should be
pleasant to read in Chinese. This module applies conservative best-effort name
localization plus a small public-copy cleanup pass to reasoning strings only.
"""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
NAME_LOCALIZATION_JSON = ROOT / "data" / "i18n" / "world_cup_2026_names.zh.json"

REASONING_KEYS = {
    "overall",
    "market_odds",
    "lineup_analysis",
    "tactical_analysis",
    "h2h_recent_form",
    "player_matchups",
    "injuries_availability",
    "upset_draw_blowout_cases",
    "score_result_rationale",
    "t1_result",
    "t2_player",
    "t3_events",
    "t4_stats",
}

TEAM_ALIASES = {
    "cabo verde": "cape verde",
    "cape verde islands": "cape verde",
}

SINGLE_TOKEN_ALLOW = {
    "Nico",
    "Rodri",
    "Pedri",
    "Gavi",
    "Yamal",
    "Vozinha",
    "Stopira",
}

SINGLE_TOKEN_STOP = {
    "United",
    "City",
    "Real",
    "Club",
    "Sporting",
    "Costa",
    "Rica",
    "Junior",
    "Senior",
    "National",
}

CODE_SPAN_RE = re.compile(r"`[^`]*`")
LATIN_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9][A-Za-zÀ-ÖØ-öø-ÿ0-9'’.-]*")
RESULT_PROBS_RE = re.compile(
    r"\bhome\s*=\s*(\d+(?:\.\d+)?)\s*[,，、/;；\s]+"
    r"draw\s*=\s*(\d+(?:\.\d+)?)\s*[,，、/;；\s]+"
    r"away\s*=\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
FIELD_VALUE_RE = re.compile(
    r"`?\b("
    r"headline_score|predicted_result|expected_total_goals|expected_goal_diff|"
    r"match_profile|most_likely_score"
    r")\b`?\s*[=:：]\s*`?([A-Za-z0-9_.+-]+)`?",
    re.IGNORECASE,
)

FIELD_LABELS = {
    "reasoning": "公开分析",
    "reasoning.overall": "整体分析",
    "market_odds": "赔率分析",
    "lineup_analysis": "阵容分析",
    "tactical_analysis": "战术分析",
    "h2h_recent_form": "交手与近况分析",
    "player_matchups": "球员对位分析",
    "injuries_availability": "伤停与可用性分析",
    "upset_draw_blowout_cases": "冷门、平局与大胜路径",
    "score_result_rationale": "比分与赛果逻辑",
    "t1_result": "赛果判断",
    "t2_player": "球员判断",
    "t3_events": "事件判断",
    "t4_stats": "数据判断",
    "predicted_result": "赛果判断",
    "headline_score": "最终比分",
    "win_probs": "胜平负概率",
    "expected_total_goals": "预期总进球",
    "expected_goal_diff": "预期净胜球",
    "match_profile": "比赛节奏",
    "score_dist": "比分分布",
    "most_likely_score": "最可能比分",
    "over_under_probs": "大小球概率",
    "expected metrics": "预期数据",
    "context_pack.odds": "结构化赔率资料",
}

PROFILE_LABELS = {
    "low_event": "低事件、偏谨慎",
    "normal": "常规节奏",
    "open": "开放对攻",
    "chaos": "混沌高事件",
}

RESULT_LABELS = {
    "home": "主队取胜",
    "draw": "双方战平",
    "away": "客队取胜",
}


def _strip_accents(value: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(ch)
    )


def _strip_parenthetical(value: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*", " ", value).strip()


def _norm_name(value: Any) -> str:
    text = _strip_parenthetical(str(value or ""))
    text = _strip_accents(text).casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = " ".join(text.split())
    return TEAM_ALIASES.get(text, text)


def _tokens(value: str) -> list[str]:
    return LATIN_TOKEN_RE.findall(str(value or ""))


def _number_key(value: Any) -> str | None:
    if value is None or value == "":
        return None
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value).strip()


def _clean_player_zh(value: Any) -> str:
    return re.sub(r"（队长）$", "", str(value or "").strip())


@lru_cache(maxsize=1)
def _load_localization() -> dict[str, Any]:
    exact: dict[str, str] = {}
    norm: dict[str, str] = {}
    teams: dict[str, str] = {}
    players_by_team_number: dict[tuple[str, str], str] = {}
    aliases_by_team_number: dict[tuple[str, str], list[str]] = {}

    if not NAME_LOCALIZATION_JSON.exists():
        return {
            "exact": exact,
            "norm": norm,
            "teams": teams,
            "players_by_team_number": players_by_team_number,
            "aliases_by_team_number": aliases_by_team_number,
        }

    raw = json.loads(NAME_LOCALIZATION_JSON.read_text())

    for source_name, item in (raw.get("teams") or {}).items():
        if not isinstance(item, dict):
            continue
        zh = str(item.get("zh") or "").strip()
        if not zh:
            continue
        aliases = {source_name, item.get("name"), _strip_parenthetical(str(source_name))}
        if _norm_name(source_name) == "cape verde":
            aliases.update({"Cape Verde Islands", "Cabo Verde"})
        for alias in aliases:
            if not alias:
                continue
            exact[str(alias)] = zh
            teams[_norm_name(alias)] = zh

    for source_name, item in (raw.get("players") or {}).items():
        if not isinstance(item, dict):
            continue
        zh = _clean_player_zh(item.get("zh"))
        if not zh:
            continue
        aliases = {
            source_name,
            item.get("name"),
            _strip_parenthetical(str(source_name)),
            _strip_parenthetical(str(item.get("name") or "")),
        }
        for alias in aliases:
            if not alias:
                continue
            exact[str(alias)] = zh
            norm[_norm_name(alias)] = zh

        team = item.get("team")
        number = _number_key(item.get("number"))
        if team and number:
            players_by_team_number[(_norm_name(team), number)] = zh
            aliases_by_team_number[(_norm_name(team), number)] = sorted(
                {str(alias).strip() for alias in aliases if alias},
                key=len,
                reverse=True,
            )

    return {
        "exact": exact,
        "norm": norm,
        "teams": teams,
        "players_by_team_number": players_by_team_number,
        "aliases_by_team_number": aliases_by_team_number,
    }


def _fixture_team_name(fixture: dict[str, Any] | None, side: str) -> str | None:
    if not isinstance(fixture, dict):
        return None

    side_value = fixture.get(side)
    if isinstance(side_value, dict):
        return (
            side_value.get("name")
            or side_value.get("team_name")
            or side_value.get("short_name")
        )
    if isinstance(side_value, str):
        return side_value

    response = fixture.get("response")
    if isinstance(response, list) and response:
        team = (((response[0].get("teams") or {}).get(side)) or {})
        if isinstance(team, dict):
            return team.get("name")
    return None


def _add_alias(mapping: dict[str, str], alias: Any, zh: str) -> None:
    alias_text = str(alias or "").strip()
    if not alias_text or not zh or alias_text == zh:
        return
    if not LATIN_TOKEN_RE.search(alias_text):
        return
    mapping.setdefault(alias_text, zh)


def _build_replacement_aliases(prediction: dict[str, Any], fixture: dict[str, Any] | None) -> dict[str, str]:
    loc = _load_localization()
    aliases: dict[str, str] = dict(loc["exact"])

    home_name = _fixture_team_name(fixture, "home")
    away_name = _fixture_team_name(fixture, "away")
    team_lookup = {
        "home": _norm_name(home_name),
        "away": _norm_name(away_name),
    }

    reasoning = prediction.get("reasoning") or {}

    # Add starting lineup aliases with team/number context when available.
    lineups = prediction.get("lineups") or {}
    for side, lineup in lineups.items():
        if side not in team_lookup:
            continue
        norm_team = team_lookup[side]
        starting = (lineup or {}).get("starting") or []
        if isinstance(lineup, list):
            starting = lineup
        for idx, player in enumerate(starting, start=1):
            if isinstance(player, dict):
                raw_name = player.get("name") or player.get("player")
                number = _number_key(player.get("number") or idx)
            else:
                raw_name = str(player)
                number = str(idx)
            if not raw_name:
                continue
            zh = loc["players_by_team_number"].get((norm_team, number))
            if zh:
                _add_alias(aliases, raw_name, zh)
                for alias in loc["aliases_by_team_number"].get((norm_team, number), []):
                    _add_alias(aliases, alias, zh)

    # Add scorer/assist/card/etc names by exact/norm lookup.
    def add_player_name(name: Any) -> None:
        raw = str(name or "").strip()
        if not raw:
            return
        zh = loc["exact"].get(raw) or loc["norm"].get(_norm_name(raw))
        if zh:
            _add_alias(aliases, raw, zh)

    for key in ("scorers", "assisters", "cards", "motm_probs"):
        for item in prediction.get(key) or []:
            if isinstance(item, dict):
                add_player_name(item.get("player"))
                add_player_name(item.get("taker"))
                add_player_name(item.get("off"))
                add_player_name(item.get("on"))

    for item in prediction.get("substitutions") or []:
        if isinstance(item, dict):
            add_player_name(item.get("off"))
            add_player_name(item.get("on"))

    for item in prediction.get("penalties") or []:
        if isinstance(item, dict):
            add_player_name(item.get("taker"))

    for item in prediction.get("own_goals") or []:
        if isinstance(item, dict):
            add_player_name(item.get("player"))

    # Team aliases for fixture participants.
    for side_name in (home_name, away_name):
        if not side_name:
            continue
        zh = loc["teams"].get(_norm_name(side_name))
        if zh:
            _add_alias(aliases, side_name, zh)
            if _norm_name(side_name) == "cape verde":
                _add_alias(aliases, "Cape Verde Islands", zh)
                _add_alias(aliases, "Cabo Verde", zh)

    return aliases


def _replace_names_in_text(text: str, aliases: dict[str, str]) -> str:
    if not text or not aliases:
        return text

    # Protect inline code spans.
    code_spans: list[str] = []
    def stash(match: re.Match[str]) -> str:
        code_spans.append(match.group(0))
        return f"￰{len(code_spans) - 1}￱"

    protected = CODE_SPAN_RE.sub(stash, text)

    # Prefer longer aliases first to avoid partial overlaps.
    for alias, zh in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = re.compile(rf"(?<![A-Za-zÀ-ÖØ-öø-ÿ0-9]){re.escape(alias)}(?![A-Za-zÀ-ÖØ-öø-ÿ0-9])")
        protected = pattern.sub(zh, protected)

    def restore(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        return code_spans[idx]

    return re.sub(r"￰(\d+)￱", restore, protected)


def _rewrite_probabilities(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        home = float(match.group(1))
        draw = float(match.group(2))
        away = float(match.group(3))
        def pct(x: float) -> str:
            v = x * 100
            if v >= 20:
                return f"约{round(v):.0f}%"
            if v >= 10:
                return f"{v:.1f}%左右"
            return f"不到{round(v):.0f}%"
        return f"主胜{pct(home)}、平局{pct(draw)}、客胜{pct(away)}"

    return RESULT_PROBS_RE.sub(repl, text)


def _rewrite_field_values(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        field = match.group(1)
        raw = match.group(2)
        label = FIELD_LABELS.get(field, field)
        if field == "predicted_result":
            value = RESULT_LABELS.get(raw.lower(), raw)
        elif field == "match_profile":
            value = PROFILE_LABELS.get(raw.lower(), raw)
        else:
            value = raw
        return f"{label}{value}"

    return FIELD_VALUE_RE.sub(repl, text)


def _cleanup_public_copy(text: str) -> str:
    if not text:
        return text
    out = text
    out = _rewrite_probabilities(out)
    out = _rewrite_field_values(out)
    return out


def localize_prediction_reasoning(
    prediction: dict[str, Any],
    *,
    fixture: dict[str, Any] | None = None,
    copy_prediction: bool = False,
) -> dict[str, Any]:
    pred = copy.deepcopy(prediction) if copy_prediction else prediction
    reasoning = pred.get("reasoning")
    if not isinstance(reasoning, dict):
        return pred

    aliases = _build_replacement_aliases(pred, fixture)
    for key, value in list(reasoning.items()):
        if key not in REASONING_KEYS or not isinstance(value, str):
            continue
        text = _replace_names_in_text(value, aliases)
        text = _cleanup_public_copy(text)
        reasoning[key] = text
    return pred

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from flask import Flask, abort, jsonify, render_template, request, send_file

DISPLAY_LIMIT = 25
BASE_DIR = Path(__file__).resolve().parent
SOLUTIONS_DIR = BASE_DIR / "solutions"
ASSETS_DIR = BASE_DIR / "assets"
ALLOWED_ASSET_TYPES = {"cards", "traits"}
MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["JSON_SORT_KEYS"] = False


@dataclass(frozen=True)
class SolutionData:
    identifier: str
    traits: Tuple[Tuple[str, int], ...]
    cards: Tuple[str, ...]


def _sorted_unique(values: Iterable[str]) -> List[str]:
    seen: Dict[str, str] = {}
    for value in values:
        key = value.lower()
        if key not in seen:
            seen[key] = value
    return sorted(seen.values(), key=str.lower)


def _parse_query_list(values: Sequence[str]) -> List[str]:
    parsed: List[str] = []
    for raw in values:
        if not raw:
            continue
        if "," in raw:
            parsed.extend(item.strip() for item in raw.split(",") if item.strip())
        else:
            stripped = raw.strip()
            if stripped:
                parsed.append(stripped)
    return parsed


@lru_cache(maxsize=1)
def _scan_solution_files() -> Dict[Tuple[int, int, str], str]:
    index: Dict[Tuple[int, int, str], str] = {}
    if not SOLUTIONS_DIR.exists():
        return index

    pattern = re.compile(r"^solutions_(\d+)_(\d+)(?:_(4trait|44trait))?\.txt$")
    for path in SOLUTIONS_DIR.glob("solutions_*.txt"):
        match = pattern.match(path.name)
        if not match:
            continue
        size = int(match.group(1))
        trait_value = int(match.group(2))
        suffix = match.group(3)
        option = "base"
        if suffix == "4trait":
            option = "4trait"
        elif suffix == "44trait":
            option = "44trait"
        index[(size, trait_value, option)] = str(path)
    return index


def _collect_assets(
    index: Dict[Tuple[int, int, str], str], deck_size: int | None = None
) -> Tuple[List[str], List[str]]:
    card_set = set()
    trait_set = set()
    for (size, _, _), path_str in index.items():
        if deck_size is not None and size != deck_size:
            continue
        for solution in _load_solutions(path_str):
            card_set.update(solution.cards)
            trait_set.update(name for name, _ in solution.traits)

    cards = sorted(card_set, key=str.lower)
    traits = sorted(trait_set, key=str.lower)
    return cards, traits


@lru_cache(maxsize=64)
def _load_solutions(file_path: str) -> Tuple[SolutionData, ...]:
    path = Path(file_path)
    if not path.exists():
        return tuple()

    raw_lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    solutions: List[SolutionData] = []

    for i in range(0, len(raw_lines), 3):
        block = raw_lines[i : i + 3]
        if len(block) < 3:
            continue
        id_line, trait_line, card_line = block

        identifier = id_line.replace("Solution nr", "").replace(":", "").strip()

        traits: List[Tuple[str, int]] = []
        for trait_entry in trait_line.split(","):
            trait_entry = trait_entry.strip()
            match = re.match(r"([a-zA-Z_ ]+)\s*\((\d+)\)", trait_entry)
            if not match:
                continue
            name = match.group(1).strip()
            count = int(match.group(2))
            traits.append((name, count))

        cards = [card.strip() for card in card_line.split(",") if card.strip()]

        if traits and cards:
            solutions.append(
                SolutionData(
                    identifier=identifier,
                    traits=tuple(traits),
                    cards=tuple(cards),
                )
            )

    return tuple(solutions)


def _calculate_trait_counts(solution: SolutionData) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name, value in solution.traits:
        counts[name.lower()] = value
    return counts


def _apply_filters(
    solutions: Sequence[SolutionData],
    required_cards: Sequence[str],
    forbidden_cards: Sequence[str],
    traits_min2: Sequence[str],
    traits_min4: Sequence[str],
    traits_max1: Sequence[str],
) -> List[SolutionData]:
    req_cards = {card.lower() for card in required_cards}
    forbid_cards = {card.lower() for card in forbidden_cards}
    req_traits2 = {trait.lower() for trait in traits_min2}
    req_traits4 = {trait.lower() for trait in traits_min4}
    req_traits2 -= req_traits4
    cap_traits = {trait.lower() for trait in traits_max1}

    filtered: List[SolutionData] = []
    for solution in solutions:
        cards_lower = {card.lower() for card in solution.cards}
        if forbid_cards & cards_lower:
            continue
        if not req_cards.issubset(cards_lower):
            continue

        trait_counts = _calculate_trait_counts(solution)

        if any(trait_counts.get(name, 0) < 4 for name in req_traits4):
            continue
        if any(trait_counts.get(name, 0) < 2 for name in req_traits2):
            continue
        if any(trait_counts.get(name, 0) >= 2 for name in cap_traits):
            continue

        filtered.append(solution)
    return filtered


@lru_cache(maxsize=512)
def _find_asset(asset_type: str, name: str) -> Path | None:
    clean = name.strip().lower()
    if not clean:
        return None
    for ext in (".webp", ".png", ".jpg", ".jpeg"):
        candidate = ASSETS_DIR / asset_type / f"{clean}{ext}"
        if candidate.exists():
            return candidate
    return None


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/metadata")
def api_metadata():
    index = _scan_solution_files()
    deck_sizes = sorted({size for size, _, _ in index.keys()})

    trait_values_by_size: Dict[str, List[int]] = {}
    option_availability: Dict[str, Dict[str, Dict[str, bool]]] = {}

    for size in deck_sizes:
        trait_values = sorted({trait for (s, trait, _) in index.keys() if s == size})
        size_key = str(size)
        trait_values_by_size[size_key] = trait_values
        option_availability[size_key] = {}
        for trait in trait_values:
            trait_key = str(trait)
            options = {
                "base": (size, trait, "base") in index,
                "4trait": (size, trait, "4trait") in index,
                "44trait": (size, trait, "44trait") in index,
            }
            option_availability[size_key][trait_key] = options

    if deck_sizes:
        default_size = 6 if 6 in deck_sizes else deck_sizes[0]
        default_trait_values = trait_values_by_size.get(str(default_size), [])
        default_trait_value = default_trait_values[-1] if default_trait_values else None
        default_option = "base"
        if default_trait_value is not None:
            options = option_availability.get(str(default_size), {}).get(str(default_trait_value), {})
            if options.get("base"):
                default_option = "base"
            elif options.get("4trait"):
                default_option = "4trait"
            elif options.get("44trait"):
                default_option = "44trait"
            else:
                default_option = "base"
    else:
        default_size = None
        default_trait_value = None
        default_option = "base"

    return jsonify(
        {
            "deckSizes": deck_sizes,
            "traitValuesBySize": trait_values_by_size,
            "optionAvailability": option_availability,
            "defaultDeckSize": default_size,
            "defaultTraitValue": default_trait_value,
            "defaultTraitOption": default_option,
            "limit": DISPLAY_LIMIT,
            "hasData": bool(index),
        }
    )


@app.get("/api/options")
def api_options():
    try:
        deck_size = int(request.args.get("deck_size"))
        traits_value = int(request.args.get("traits_value"))
    except (TypeError, ValueError):
        abort(400, description="Missing or invalid deck_size or traits_value parameter")

    trait_option = request.args.get("trait_option", "base").lower()
    if trait_option not in {"base", "4trait", "44trait"}:
        abort(400, description="Invalid trait_option parameter")

    index = _scan_solution_files()
    file_path = index.get((deck_size, traits_value, trait_option))
    if not file_path:
        abort(404, description="No solutions available for the requested combination")

    solutions = _load_solutions(file_path)
    cards = _sorted_unique(card for sol in solutions for card in sol.cards)
    traits = _sorted_unique(trait for sol in solutions for trait, _ in sol.traits)

    if not cards or not traits:
        deck_cards, deck_traits = _collect_assets(index, deck_size)
        if not cards:
            cards = deck_cards
        if not traits:
            traits = deck_traits

    if not cards or not traits:
        all_cards, all_traits = _collect_assets(index, None)
        if not cards:
            cards = all_cards
        if not traits:
            traits = all_traits

    return jsonify({"cards": cards, "traits": traits})


@app.get("/api/solutions")
def api_solutions():
    try:
        deck_size = int(request.args.get("deck_size"))
        traits_value = int(request.args.get("traits_value"))
    except (TypeError, ValueError):
        abort(400, description="Missing or invalid deck_size or traits_value parameter")

    trait_option = request.args.get("trait_option", "base").lower()
    if trait_option not in {"base", "4trait", "44trait"}:
        abort(400, description="Invalid trait_option parameter")

    index = _scan_solution_files()
    file_path = index.get((deck_size, traits_value, trait_option))
    if not file_path:
        abort(404, description="No solutions available for the requested combination")

    solutions = _load_solutions(file_path)

    required_cards = _parse_query_list(request.args.getlist("required_cards"))
    forbidden_cards = _parse_query_list(request.args.getlist("forbidden_cards"))
    traits_min4 = _parse_query_list(request.args.getlist("traits_min4"))
    traits_min2 = _parse_query_list(request.args.getlist("traits_min2"))
    traits_max1 = _parse_query_list(request.args.getlist("traits_max1"))

    filtered = _apply_filters(solutions, required_cards, forbidden_cards, traits_min2, traits_min4, traits_max1)

    try:
        limit = int(request.args.get("limit", DISPLAY_LIMIT))
    except ValueError:
        limit = DISPLAY_LIMIT
    limit = max(1, min(limit, 100))

    total_matches = len(filtered)
    total_pages = math.ceil(total_matches / limit) if total_matches else 0

    try:
        requested_page = int(request.args.get("page", 1))
    except ValueError:
        requested_page = 1

    if total_pages:
        page = max(1, min(requested_page, total_pages))
    else:
        page = 1

    start = (page - 1) * limit if total_matches else 0
    end = min(start + limit, total_matches)
    page_items = filtered[start:end] if total_matches else []

    results = []
    for offset, item in enumerate(page_items):
        display_id = start + offset + 1 if total_matches else offset + 1
        results.append(
            {
                "displayId": display_id,
                "identifier": item.identifier,
                "cards": list(item.cards),
                "traits": [
                    {"name": name, "count": count} for name, count in item.traits
                ],
            }
        )

    return jsonify(
        {
            "page": page,
            "limit": limit,
            "totalMatches": total_matches,
            "totalPages": total_pages,
            "results": results,
        }
    )


@app.get("/assets/logo.png")
def serve_logo():
    path = ASSETS_DIR / "logo3.png"
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="image/png")


@app.get("/assets/merge_tactics_logo.webp")
def serve_seasonal_logo():
    path = ASSETS_DIR / "merge_tactics_logo.webp"
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="image/webp")


@app.get("/assets/header_backround.png")
def serve_header_background():
    path = ASSETS_DIR / "header_backround.png"
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="image/png")


@app.get("/assets/<asset_type>")
def serve_asset(asset_type: str):
    if asset_type not in ALLOWED_ASSET_TYPES:
        abort(404)

    name = request.args.get("name", "")
    path = _find_asset(asset_type, name)
    if not path:
        abort(404)
    mimetype = MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return send_file(path, mimetype=mimetype)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

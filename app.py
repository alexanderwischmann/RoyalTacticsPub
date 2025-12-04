from __future__ import annotations

import math
import re
from collections import deque
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
import json

from flask import Flask, abort, jsonify, render_template, request, send_file

DISPLAY_LIMIT = 25
# How often (in iterations) to check for cancellation to avoid per-iteration
# locking overhead. Increase to check less frequently (lower CPU cost), or
# decrease to respond to cancellations faster.
CANCELLATION_CHECK_INTERVAL = 128
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

with open(f"{BASE_DIR}/card_dict.json", "r", encoding="utf-8") as f:
    CARD_DICT = json.load(f)
CARD_LIST = sorted(CARD_DICT.keys())
TRAIT_LIST = sorted(set(trait for traits in CARD_DICT.values() for trait in traits))

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["JSON_SORT_KEYS"] = False


@dataclass(frozen=True)
class SolutioMask:
    cards: int # Bitmask
    traits:  int # Bitmask

@dataclass(frozen=True)
class SolutionData:
    identifier: int
    traits: Tuple[Tuple[str, int], ...]
    cards: Tuple[str, ...]

# --------- BITFILTER LOGIC  ----------

def traitsMask(weighted_traits: list[str]) -> int:
    mask = 0
    for trait, weight in weighted_traits: 
        weight_mask = 0
        for weight_index in range(weight):
            weight_mask |= (1 << weight_index) # for weight 1 to 4:  0001 0011 0111 1111
        
        index = TRAIT_LIST.index(trait)
        mask |= (weight_mask << (index * 4))
    return mask

def cardsMask(cards: list[str]) -> int:
    mask = 0
    for card in cards:
        index = CARD_LIST.index(card)
        mask |= (1 << index)
    return mask


def filterCardsForbidden(filterMask: int, cardsMask: int) -> bool:
    return (filterMask & cardsMask) == 0

def filterCardsRequired(filterMask: int, cardsMask: int) -> bool:
    return (filterMask & cardsMask) == filterMask

def filterTraitsMax1(filterMask: int, traitsMask: int) -> bool:
    return (filterMask & traitsMask) == 0

def filterTraitsMin2(filterMask: int, traitsMask: int) -> bool:
    return (filterMask & traitsMask) == filterMask

def filterTraitsMin4(filterMask: int, traitsMask: int) -> bool:
    return (filterMask & traitsMask) == filterMask


def cardsRequiredFilterMask(requiredCards: list[str]) -> int:
    return cardsMask(requiredCards)

def cardsForbiddenFilterMask(forbiddenCards: list[str]) -> int:
    return cardsMask(forbiddenCards)

def traitsMax1FilterMask(max1Traits: list[str]) -> int:
    mask = 0
    for trait in max1Traits:
        index = TRAIT_LIST.index(trait)
        mask |= (1 << (index * 4))
    return mask

def traitMin2FilterMask(min2Traits: list[str]) -> int:
    mask = 0
    for trait in min2Traits:
        index = TRAIT_LIST.index(trait)
        weight_mask = (1 << 1) | (1 << 2) # (0011 & x = 0011)  true for 0011, 0111, 1111  (i.e. weight >= 2)
        mask |= (weight_mask << (index * 4))
    return mask

def traitMin4FilterMask(min4Traits: list[str]) -> int:
    mask = 0
    for trait in min4Traits:
        index = TRAIT_LIST.index(trait)
        weight_mask = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) # (1111 & x = 1111)  true for 1111 only (i.e. weight == 4)
        mask |= (weight_mask << (index * 4))
    return mask

def decodeTraitsMask(traitsMask: int) -> List[Tuple[str, int]]:
    traits = []
    for index, trait in enumerate(TRAIT_LIST):
        weight_mask = (traitsMask >> (index * 4)) & 0b1111
        if weight_mask != 0:
            weight = 0
            for bit_index in range(4):
                if (weight_mask >> bit_index) & 1:
                    weight += 1
            traits.append((trait, weight))
    return traits

def decodeCardsMask(cardsMask: int) -> List[str]:
    cards = []
    for index, card in enumerate(CARD_LIST):
        if (cardsMask >> index) & 1:
            cards.append(card)
    return cards

# --------- END BITFILTER LOGIC  ----------


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



def _iter_solutions(file_path: str) -> Iterator[SolutioMask]:
    path = Path(file_path)
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(",")
            yield SolutioMask(
                cards=int(parts[0]),
                traits=int(parts[1]),
            )


def _solution_matches_filters(
    solution: SolutioMask,
    requiredCardsMask: int,
    forbiddenCardsMask: int,
    traitsMax1Mask: int,
    traitsMin2Mask: int,
    traitsMin4Mask: int,
) -> bool:
    
    return (
        filterCardsRequired(requiredCardsMask, solution.cards) and
        filterCardsForbidden(forbiddenCardsMask, solution.cards) and
        filterTraitsMax1(traitsMax1Mask, solution.traits) and
        filterTraitsMin2(traitsMin2Mask, solution.traits) and
        filterTraitsMin4(traitsMin4Mask, solution.traits) 
)

def _decode_solutions(solutions: List[int, SolutioMask]) -> List[SolutionData]:
    return [SolutionData(
        identifier=index,
        traits=tuple(decodeTraitsMask(solution.traits)),
        cards=tuple(decodeCardsMask(solution.cards))
    ) for index, solution in solutions]

def _collect_filtered_page(
    file_path: str,
    start_index: int,
    limit: int,
    required_cards: Set[str],
    forbidden_cards: Set[str],
    traits_min2: Set[str],
    traits_min4: Set[str],
    traits_max1: Set[str],
    client_ip: str,
    token: int,
) -> Tuple[int, List[SolutionData], List[SolutionData], bool]:
    total_matches = 0
    requested_entries_masked: List[Tuple[int, SolutioMask]] = []
    tail_entries_masked: deque[Tuple[int, SolutioMask]] = deque(maxlen=limit)

    # Precompute filter masks
    requiredCardsMask = cardsRequiredFilterMask(list(required_cards))
    forbiddenCardsMask = cardsForbiddenFilterMask(list(forbidden_cards))
    traitsMax1Mask = traitsMax1FilterMask(list(traits_max1))
    traitsMin2Mask = traitMin2FilterMask(list(traits_min2))
    traitsMin4Mask = traitMin4FilterMask(list(traits_min4))

    # To avoid the overhead of acquiring a lock on every iteration we check
    # the cancellation token only every `CANCELLATION_CHECK_INTERVAL`
    # iterations. We perform a plain dict read (no lock) for the quick-path
    # check — this is safe on CPython (single dict access is atomic) and the
    # write-side always uses REQUEST_LOCK so we won't observe a crash; at
    # worst we'll see a slightly stale value for a short time which is
    # acceptable for cancellation semantics.
    iterations_since_check = 0
    for solution in _iter_solutions(file_path):
        iterations_since_check += 1
        if iterations_since_check >= CANCELLATION_CHECK_INTERVAL:
            iterations_since_check = 0
            # read without lock for performance; use the locked path only if
            # you need a strict memory-model guarantee across Python
            current = REQUEST_TOKENS.get(client_ip, 0)
            if current != token:
                return (0, [], [], True)
        if not _solution_matches_filters(
            solution,
            requiredCardsMask,
            forbiddenCardsMask,
            traitsMax1Mask,
            traitsMin2Mask,
            traitsMin4Mask,
        ):
            continue

        match_index = total_matches
        total_matches += 1

        if start_index <= match_index < start_index + limit:
            requested_entries_masked.append((match_index, solution))

        tail_entries_masked.append((match_index, solution))

    return (
        total_matches,
        _decode_solutions(requested_entries_masked),
        _decode_solutions(list(tail_entries_masked)),
        False,
    )


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


# Map of client_ip -> latest token (int). When a new request arrives we bump the
# token so any earlier in-flight processing can detect the mismatch and stop.
REQUEST_TOKENS: Dict[str, int] = {}
REQUEST_LOCK = threading.Lock()


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

    return jsonify({"cards": CARD_LIST, "traits": TRAIT_LIST})


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

    required_cards_raw = _parse_query_list(request.args.getlist("required_cards"))
    forbidden_cards_raw = _parse_query_list(request.args.getlist("forbidden_cards"))
    traits_min4_raw = _parse_query_list(request.args.getlist("traits_min4"))
    traits_min2_raw = _parse_query_list(request.args.getlist("traits_min2"))
    traits_max1_raw = _parse_query_list(request.args.getlist("traits_max1"))

    required_cards = {card.lower() for card in required_cards_raw}
    forbidden_cards = {card.lower() for card in forbidden_cards_raw}
    traits_min4 = {trait.lower() for trait in traits_min4_raw}
    traits_min2 = {trait.lower() for trait in traits_min2_raw}
    traits_min2 -= traits_min4
    traits_max1 = {trait.lower() for trait in traits_max1_raw}

    try:
        limit = int(request.args.get("limit", DISPLAY_LIMIT))
    except ValueError:
        limit = DISPLAY_LIMIT
    limit = max(1, min(limit, 100))

    try:
        requested_page = int(request.args.get("page", 1))
    except ValueError:
        requested_page = 1
    requested_page = max(1, requested_page)

    start_index = (requested_page - 1) * limit

    # client identification: prefer client-supplied `clientId` and `requestId`.
    # If absent, fall back to IP-based behavior for compatibility.
    client_id_arg = request.args.get("clientId")
    request_id_arg = request.args.get("requestId")

    if client_id_arg:
        client_key = client_id_arg
    else:
        client_key = request.remote_addr or request.environ.get("REMOTE_ADDR") or "unknown"

    # Determine request id: prefer client-supplied increasing integer. If not
    # provided, fall back to bumping the stored token (older behavior).
    if request_id_arg is not None:
        try:
            request_id = int(request_id_arg)
        except ValueError:
            abort(400, description="Invalid requestId parameter")
        # Record the latest seen request id for this client (use max to be robust
        # against out-of-order arrivals).
        with REQUEST_LOCK:
            prev = REQUEST_TOKENS.get(client_key, 0)
            if request_id > prev:
                REQUEST_TOKENS[client_key] = request_id
    else:
        # Older clients: bump token so in-flight work is invalidated.
        with REQUEST_LOCK:
            request_id = REQUEST_TOKENS.get(client_key, 0) + 1
            REQUEST_TOKENS[client_key] = request_id

    total_matches, requested_entries, tail_entries, cancelled = _collect_filtered_page(
        file_path,
        start_index,
        limit,
        required_cards,
        forbidden_cards,
        traits_min2,
        traits_min4,
        traits_max1,
        client_key,
        request_id,
    )

    if cancelled:
        return jsonify({"cancelled": True}), 200

    if total_matches == 0:
        total_pages = 0
        page = 1
        entries: List[SolutionData] = []
    else:
        total_pages = math.ceil(total_matches / limit)
        page = min(requested_page, total_pages)
        if page == requested_page:
            entries = requested_entries
        else:
            adjusted_start = (page - 1) * limit
            entries = [entry for entry in tail_entries if entry.identifier >= adjusted_start]

    results = [
        {
            "displayId": solution.identifier + 1,
            "cards": list(solution.cards),
            "traits": [
                {"name": name, "count": count} for name, count in solution.traits
            ],
        }
        for solution in entries
    ]

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

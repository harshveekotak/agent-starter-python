import re

COMBO_SEPARATORS = ["+", "&", "and"]

def split_combo_title(title: str) -> list[str]:
    """
    Splits a combo title into service names.
    Handles '+', '&', 'and'.
    """
    normalized = title.lower()

    for sep in COMBO_SEPARATORS:
        normalized = normalized.replace(sep, "|")

    parts = [p.strip().title() for p in normalized.split("|")]
    return [p for p in parts if p]


def extract_combo_addons(combo_title: str, base_service: str) -> list[str]:
    """
    Returns only the additional services, excluding the base service.
    """
    parts = split_combo_title(combo_title)
    base_lower = base_service.lower()

    return [
        p for p in parts
        if base_lower not in p.lower()
    ]

def combo_score(title: str) -> int:
    title = title.lower()
    score = 0
    if "combo" in title:
        score += 3
    if "package" in title:
        score += 2
    if "+" in title or "&" in title:
        score += 1
    return score
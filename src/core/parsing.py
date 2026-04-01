def extract_index_from_user_input(text: str) -> int | None:
    if not text:
        return None

    text = text.lower().strip()

    if "first" in text:
        return 0
    if "second" in text:
        return 1
    if "third" in text:
        return 2

    for token in text.split():
        if token.isdigit():
            idx = int(token) - 1
            if idx >= 0:
                return idx

    return None

from core.logger import log_event

ALLOWED_INTENTS = {
    "greeting",
    "thanks",
    "goodbye",

    "service_selection",
    "combo_inquiry",
    "recommendation_request",

    "check_availability",
    "book_appointment",
    "reschedule_appointment",
    "cancel_appointment",

    "confirm",
    "reject",
}

def guard_intent(intent: str, text: str, session) -> str | None:
    """
    Returns a response string if intent is NOT allowed.
    Return None if the intent is allowed.
    """

    if intent in ALLOWED_INTENTS:
        return None

    return ("I'm here specifically to help with booking services like "
            "haircuts, spa, or massage appointments. "
            "What would you like to book today?")
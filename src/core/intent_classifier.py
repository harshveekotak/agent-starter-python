"""
Semantic intent classifier using sentence embeddings.
Handles yes/no, cancel intent, and reschedule intent.
Replaces unreliable zero-shot bart confidence thresholds.
"""
from sentence_transformers import SentenceTransformer, util

_model = SentenceTransformer("all-MiniLM-L6-v2")

# ── YES / NO ──────────────────────────────────────────────────────────────────
_YES_ANCHORS = [
    "yes", "yeah", "confirmed", "go ahead", "book it",
    "that works", "sounds good", "do it", "correct", "please proceed",
]
_NO_ANCHORS = [
    "no", "nope", "don't book", "not that",
    "that doesn't work", "stop", "wrong", "skip it", "never mind",
]

_yes_vecs = _model.encode(_YES_ANCHORS, convert_to_tensor=True)
_no_vecs  = _model.encode(_NO_ANCHORS,  convert_to_tensor=True)


def is_confirm(text: str, threshold: float = 0.45) -> bool:
    vec = _model.encode(text, convert_to_tensor=True)
    yes_score = float(util.cos_sim(vec, _yes_vecs).max())
    no_score  = float(util.cos_sim(vec, _no_vecs).max())
    return yes_score >= threshold and yes_score > no_score


def is_reject(text: str, threshold: float = 0.45) -> bool:
    vec = _model.encode(text, convert_to_tensor=True)
    no_score  = float(util.cos_sim(vec, _no_vecs).max())
    yes_score = float(util.cos_sim(vec, _yes_vecs).max())
    return no_score >= threshold and no_score > yes_score


# ── CANCEL INTENT ─────────────────────────────────────────────────────────────
_CANCEL_ANCHORS = [
    "cancel my appointment", "cancel the booking", "cancel it",
    "I want to cancel", "please cancel", "cancel this",
    "don't want the appointment", "remove my booking", "call it off",
    "actually cancel", "forget the appointment",
]
_cancel_vecs = _model.encode(_CANCEL_ANCHORS, convert_to_tensor=True)
_CANCEL_KEYWORDS = {"cancel", "cancellation", "call off", "remove booking"}


def has_cancel_keyword(text: str) -> bool:
    """Returns True if an explicit cancel keyword is present in the text.
    This is always reliable regardless of other context (date/time present etc.)."""
    text_l = text.lower()
    return any(kw in text_l for kw in _CANCEL_KEYWORDS)


def is_cancel_intent(text: str, threshold: float = 0.60) -> bool:
    """Semantic-only cancel detection (no keyword fast-path).
    Use this when no explicit keyword is present.
    Guarded against fresh booking language and date+time messages."""
    text_l = text.lower()

    # Never fire on fresh booking language
    if any(sig in text_l for sig in _FRESH_BOOKING_SIGNALS):
        return False

    # Semantic fallback with high threshold
    vec = _model.encode(text, convert_to_tensor=True)
    return float(util.cos_sim(vec, _cancel_vecs).max()) >= threshold


# ── RESCHEDULE INTENT ─────────────────────────────────────────────────────────
_RESCHEDULE_ANCHORS = [
    "reschedule my appointment", "change my booking", "move my appointment",
    "I want to reschedule", "change the time", "shift my booking",
    "book a different time", "change the date", "can we move it",
    "actually reschedule", "different time slot",
]
_reschedule_vecs = _model.encode(_RESCHEDULE_ANCHORS, convert_to_tensor=True)
_RESCHEDULE_KEYWORDS = {"reschedule", "rebook", "move my", "change my booking"}

# Words that strongly indicate a FRESH booking, not a reschedule
_FRESH_BOOKING_SIGNALS = {
    "book me", "book an", "book a", "book both", "book spa",
    "book facial", "book hair", "book massage", "book manicure",
    "book pedicure", "make me", "make an", "i want to book",
    "i need to book", "add both", "schedule me", "schedule a"
}

# Words that strongly indicate a CANCEL, not a reschedule
_CANCEL_SIGNALS = {
    "cancel", "cancellation", "call off", "remove booking",
    "cancel my", "cancel the", "cancel this", "cancel it",
    "cancel appointment", "cancel booking"
}

def is_reschedule_intent(text: str, threshold: float = 0.50) -> bool:
    """Keyword fast-path + semantic fallback for reschedule intent.
    Never fires on fresh booking or cancel language."""
    text_l = text.lower()

    # Never treat cancel messages as reschedule
    if any(sig in text_l for sig in _CANCEL_SIGNALS):
        return False

    # If it reads like a fresh booking request, it is NOT a reschedule
    if any(sig in text_l for sig in _FRESH_BOOKING_SIGNALS):
        return False

    # Fast path: explicit reschedule keyword
    if any(kw in text_l for kw in _RESCHEDULE_KEYWORDS):
        return True

    # Semantic fallback
    vec = _model.encode(text, convert_to_tensor=True)
    return float(util.cos_sim(vec, _reschedule_vecs).max()) >= threshold
"""
Keyword-based intent detection -- replaces facebook/bart-large-mnli.
No model loading, no GPU/RAM requirement, instant response.
Returns (intent_label, confidence) matching the original API.
"""

import re

INTENTS = [
    "book_appointment",
    "cancel_appointment",
    "reschedule_appointment",
    "check_availability",
    "recommendation_request",
    "thanks",
    "goodbye",
    "combo_inquiry",
]

_BOOK_WORDS = {
    "book", "booking", "schedule", "appointment", "reserve", "set up",
    "make an appointment", "get an appointment", "want a", "need a",
    "haircut", "hair spa", "massage", "facial", "manicure", "pedicure",
    "slot", "available", "fix me", "fix a", "arrange",
}

_CANCEL_WORDS = {
    "cancel", "cancellation", "delete booking", "remove booking",
    "don't want", "dont want", "call off", "abort", "stop appointment",
}

_RESCHEDULE_WORDS = {
    "reschedule", "rescheduling", "move my", "move the", "change my",
    "change the", "push", "shift", "postpone", "different time",
    "different day", "different date", "new time", "new date",
    "change time", "change date",
}

_AVAIL_WORDS = {
    "available", "availability", "open", "free slot", "free time",
    "what times", "which times", "what slots", "which slots",
    "when are you", "do you have",
}

_RECOMMEND_WORDS = {
    "recommend", "suggestion", "suggest", "what do you recommend",
    "what should i", "best option", "popular", "combo",
}

_THANKS_WORDS = {
    "thanks", "thank you", "thank u", "thx", "ty", "appreciate",
    "great", "awesome", "perfect", "wonderful",
}

_GOODBYE_WORDS = {
    "bye", "goodbye", "see you", "later", "farewell", "take care",
    "that's all", "thats all", "nothing else", "no more",
}

_COMBO_WORDS = {
    "combo", "add on", "add-on", "package", "bundle", "pair",
    "go well", "goes well", "combine",
}


def detect_intent(text: str):
    t = text.lower().strip()
    words = set(re.findall(r"[\w']+", t))

    def _hits(keyword_set):
        count = 0
        for kw in keyword_set:
            if " " in kw:
                if kw in t:
                    count += 2
            elif kw in words:
                count += 1
        return count

    scores = {
        "reschedule_appointment": _hits(_RESCHEDULE_WORDS),
        "cancel_appointment":     _hits(_CANCEL_WORDS),
        "check_availability":     _hits(_AVAIL_WORDS),
        "combo_inquiry":          _hits(_COMBO_WORDS),
        "recommendation_request": _hits(_RECOMMEND_WORDS),
        "thanks":                 _hits(_THANKS_WORDS),
        "goodbye":                _hits(_GOODBYE_WORDS),
        "book_appointment":       _hits(_BOOK_WORDS),
    }

    best = max(scores, key=lambda k: scores[k])
    best_score = scores[best]

    if best_score == 0:
        return "book_appointment", 0.5

    # Normalise to 0-1
    total = sum(scores.values()) or 1
    confidence = round(best_score / total, 3)
    return best, confidence
import re
import dateparser
from utils import record_recommendation


def parse_slot_choice(text: str):
    """
    Converts user replies into normalized time like HH:MM.

    Examples:
    "6" → "06:00"
    "6pm" → "18:00"
    "6:30" → "06:30"
    "evening 6" → "18:00"
    """

    text = text.lower().strip()

    # If user says "second one", "first one"
    if "first" in text:
        return "__FIRST__"
    if "second" in text:
        return "__SECOND__"
    if "third" in text:
        return "__THIRD__"

    # Use dateparser for flexible time understanding
    dt = dateparser.parse(text)

    if not dt:
        return None

    return dt.strftime("%H:%M")


def select_slot(session, slots):
    """
    Selects a slot and records the recommendation.
    """

    if not slots:
        return None
    

    chosen_slot = slots[0]

    record_recommendation(
        session,
        key=chosen_slot,
        rec_type="slot",
        signals={
            "requested_time": session.time,
            "requested_date": session.date,
            "total_available_slots": len(slots),
            "strategy": "earliest_available"
        }
    )

    return chosen_slot
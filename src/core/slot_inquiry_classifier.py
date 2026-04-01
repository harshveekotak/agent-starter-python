"""
Slot inquiry intent classifier.
Uses a small set of diverse training examples + embedding similarity
to detect when a user wants to SEE available slots rather than book a specific time.
"""

from core.embeddings import embed_text
from sentence_transformers import util

# ── Training examples ──────────────────────────────────────────────────────────
# These are POSITIVE examples: messages that mean "show me what's available"
# They are diverse on purpose -- the model generalises from variety, not volume.
_POSITIVE = [
    "which slots are available",
    "what times are open today",
    "show me available times",
    "when can I come in",
    "what do you have free",
    "any openings today",
    "what are my options",
    "what times can I book",
    "I want a haircut, what's available",
    "I want a trim, which slots are free",
    "give me the earliest slot",
    "what's the soonest I can come",
    "check availability for me",
    "I need a massage, when are you free",
    "do you have anything open",
    "what times work today",
    "let me see the available slots",
    "I want a pedicure, what times do you have",
    "show me times for today",
    "are there any slots left",
    "what time slots do you have",
    "I'd like a facial, when can I book",
    "any available appointments",
    "what are the open times",
    "I want to book, what's available",
]

# ── Negative examples: things that LOOK similar but are NOT slot inquiries ─────
_NEGATIVE = [
    "book me an appointment at 3pm",
    "I want a haircut at 2 tomorrow",
    "reschedule my appointment",
    "cancel my booking",
    "I want to book for Friday at 10",
    "set it for 4pm today",
    "I'd like 11am please",
    "confirm my booking",
    "I need to cancel",
    "move my appointment to Thursday",

     # ── Ordinal picks ──────────────────────────────────────────────
    "the first one",
    "the second one",
    "the third one",
    "the last one",
    "number two",
    "option 1",
    "that one",
    "this one",
    "the other one",
    # ── Same time phrases ──────────────────────────────────────────
    "same time",
    "the same time",
    "keep the same time",
    "same slot",
    "same as before",
    "at the same time as before",
    "same timing",
]

# ── Pre-compute embeddings once at import time ─────────────────────────────────
_pos_vecs = [embed_text(t) for t in _POSITIVE]
_neg_vecs = [embed_text(t) for t in _NEGATIVE]

_pos_centroid = sum(_pos_vecs) / len(_pos_vecs)
_neg_centroid = sum(_neg_vecs) / len(_neg_vecs)


def is_slot_inquiry(text: str, threshold: float = 0.15) -> bool:
    """
    Returns True if the user wants to see available time slots.

    Uses centroid-based intent classification:
      - Compute similarity to positive centroid (slot inquiry intent)
      - Compute similarity to negative centroid (booking/cancel/reschedule intent)
      - Return True if pos_score > neg_score AND pos_score >= threshold
      - Also returns True if ANY individual positive example scores very high (>= 0.75)

    This generalises to ANY phrasing the user might use --
    no need to add new examples for every variation.
    """
    if len(text.strip().split()) <= 2:
        return False
        
    vec = embed_text(text)

    pos_score = float(util.cos_sim(vec, _pos_centroid))
    neg_score = float(util.cos_sim(vec, _neg_centroid))

    # Check best individual match (catches very close paraphrases)
    best_individual = max(float(util.cos_sim(vec, pv)) for pv in _pos_vecs)

    result = (
        (pos_score > neg_score and pos_score >= threshold)
        or best_individual >= 0.75
    )

    print(f"DEBUG SLOT INQUIRY: pos={pos_score:.3f} neg={neg_score:.3f} "
          f"best_match={best_individual:.3f} → {result}")

    return result
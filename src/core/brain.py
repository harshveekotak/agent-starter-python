from core.sessions import Session
from core.extractor import extract_service, extract_all_services
from core.intent import detect_intent
from core.recommender import recommend
from core.catalog import CATALOG, COMBO_SUGGESTIONS
from core.datetime_parser import extract_datetime
from core.user_details_parser import extract_user_details
from core.slot_picker import parse_slot_choice
from core.logger import log_event
from core.storage import (
    load_latest_session, save_session, clear_session,
    load_profile, save_profile, clear_profile,
)
from core.booking_flow import run_booking_flow
from core.parsing import extract_index_from_user_input
from core.intent_classifier import is_confirm, is_reject, is_cancel_intent, is_reschedule_intent, has_cancel_keyword

def _fmt_booking(b):
    """Portable cross-platform formatter for a booking record."""
    import re as _re_fmt, pytz as _ptz_fmt
    from datetime import datetime as _dt_fmt
    try:
        _st = _re_fmt.sub(r"\.\d+Z$", "Z", b.get("start_time","")).replace(".000Z","Z").replace("Z","+00:00")
        _bs = _dt_fmt.fromisoformat(_st).astimezone(_ptz_fmt.timezone("Asia/Kolkata"))
        _svc = ", ".join(b.get("services", [])) or "Appointment"
        _hour = int(_bs.strftime("%I"))  # strip leading zero portably
        _ampm = _bs.strftime("%p")
        _min = _bs.strftime("%M")
        _time_str = f"{_hour}:{_min} {_ampm}"
        return f"{_svc} on {_bs.strftime('%b')} {_bs.day} at {_time_str}"
    except Exception:
        return b.get("start_time", "unknown time")

def _pick_booking_by_intent(user_text: str, options: list, tz) -> int | None:
    """
    Pure semantic booking picker — no hardcoded ordinal wordlists.

    Works by embedding the user's natural language query and a rich
    natural-language description of each booking that includes:
      - absolute forward position  (first, second, third...)
      - absolute reverse position  (last, second last, third last...)
      - date / time / service content
      - relational context         (before X, after X, earliest, latest)

    The embedding model handles all surface variations:
      "second last", "penultimate", "the one before the last",
      "second from the bottom", "not the final one", etc.
    """
    from core.embeddings import embed_text
    from sentence_transformers import util
    from datetime import datetime as _dt_pb
    import re as _re_pb

    if not options:
        return None

    total = len(options)

    # ------------------------------------------------------------------
    # Pre-parse all booking datetimes so relational context can be added
    # ------------------------------------------------------------------
    parsed = []
    for b in options:
        try:
            _st = _re_pb.sub(r"\.\d+Z$", "Z", b["start_time"]).replace(".000Z", "Z").replace("Z", "+00:00")
            parsed.append(_dt_pb.fromisoformat(_st).astimezone(tz))
        except Exception:
            parsed.append(None)

    # ------------------------------------------------------------------
    # Build rich natural-language descriptions
    # ------------------------------------------------------------------
    descriptions = []
    for i, (b, bs) in enumerate(zip(options, parsed)):
        parts = []

        # ── Absolute forward position ──────────────────────────────────
        fwd = {
            0: "the first booking, the earliest one, the one at the top, option 1, number one",
            1: "the second booking, option 2, number two",
            2: "the third booking, option 3, number three",
            3: "the fourth booking, option 4, number four",
            4: "the fifth booking, option 5, number five",
        }.get(i, f"option {i + 1}, number {i + 1}")
        parts.append(fwd)

        # ── Absolute reverse position ──────────────────────────────────
        rev = total - 1 - i   # 0 = last, 1 = second-last, 2 = third-last …
        if rev == 0:
            parts.append(
                "the last booking, the final one, the very last one, "
                "the most recent, at the bottom of the list, "
                "the end, the last option, not second last, not penultimate, "
                "truly the last, the absolute last, the final booking, "
                "not second last, not penultimate, not one before last, "
                "truly final, nothing comes after this"
            )
        elif rev == 1:
            parts.append(
                "the second last booking, second to last, "
                "the penultimate one, one before the last, "
                "second from the bottom, second from the end, "
                "not the last but the one before it, not the final one, "
                "the one just above the last, second from bottom, "
                "penultimate position, one step before the end, "
                "not the very last, the one right before the final one"
            )
        elif rev == 2:
            parts.append(
                "the third last booking, third to last, "
                "third from the bottom, third from the end, "
                "two before the last"
            )
        elif rev == 3:
            parts.append(
                "the fourth last booking, fourth to last, "
                "fourth from the bottom"
            )

        # ── Middle / boundary labels ───────────────────────────────────
        if i == 0 and total > 1:
            parts.append("the earliest appointment, comes first chronologically")
        if i == total - 1 and total > 1:
            parts.append("the latest appointment, comes last chronologically")
        if 0 < i < total - 1:
            parts.append("a middle option, neither the first nor the last")

        # ── Service content ────────────────────────────────────────────
        svcs = ", ".join(b.get("services", [])) or "appointment"
        parts.append(f"service is {svcs}, booking for {svcs}")

        # ── Date / time content + relational context ───────────────────
        if bs is not None:
            day_name  = bs.strftime("%A")        # "Monday"
            month_day = bs.strftime("%B %d")     # "March 25"
            hour      = int(bs.strftime("%I"))   # strip leading zero
            ampm      = bs.strftime("%p")
            minute    = bs.strftime("%M")
            parts.append(
                f"on {day_name}, {month_day}, at {hour}:{minute} {ampm}, "
                f"{day_name} appointment, {month_day} booking"
            )
            # Neighbour-relative labels
            if i > 0 and parsed[i - 1] is not None:
                prev_label = parsed[i - 1].strftime("%B %d")
                parts.append(f"after {prev_label}, comes after the {prev_label} booking")
            if i < total - 1 and parsed[i + 1] is not None:
                next_label = parsed[i + 1].strftime("%B %d")
                parts.append(f"before {next_label}, comes before the {next_label} booking")

        descriptions.append(". ".join(parts))

    # ------------------------------------------------------------------
    # Embed and score
    # ------------------------------------------------------------------
    try:
        user_vec  = embed_text(user_text)
        desc_vecs = [embed_text(d) for d in descriptions]
        scores    = [float(util.cos_sim(user_vec, dv)) for dv in desc_vecs]

        best_idx   = max(range(len(scores)), key=lambda i: scores[i])
        best_score = scores[best_idx]

        print(f"DEBUG PICK RESULT: best_idx={best_idx} score={best_score:.2f}")

        sorted_scores = sorted(scores, reverse=True)
        gap = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 1.0

        if best_score >= 0.15 and gap >= 0.005:
            return best_idx

        return None   # ambiguous
    except Exception as e:
        print(f"DEBUG PICK ERROR: {e}")
        return None


class Brain:

    def __init__(self):
        """Initialize Brain with fresh session state.

        Profile is NOT loaded here — we don't know the user's email yet.
        Profile will be loaded in _handle_text_inner after email is collected.
        """
        # Don't auto-load profile — we don't know the user's email yet
        # Profile will be loaded after email is collected in _handle_text_inner
        self.profile = {}

        # Always start fresh — never resume from disk in voice mode
        self.session = Session()

        # Always start with clean reschedule + cancel state
        # Prevents stale sessions.json state from bleeding into fresh conversation
        self.session.awaiting_reschedule = False
        self.session.awaiting_reschedule_email = False
        self.session.awaiting_reschedule_pick = False
        self.session.awaiting_reschedule_confirm = False
        self.session._reschedule_booking = None
        self.session._reschedule_options = []
        self.session._reschedule_email = None
        self.session.awaiting_cancel_email = False
        self.session.awaiting_cancel_pick = False
        self.session.awaiting_cancel_confirm = False
        self.session.awaiting_cancel_reason = False
        self.session.awaiting_cancel_retry = False

        # Add current date for slot inquiry fallback
        import pytz
        from datetime import datetime
        self.today = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d")

        self.recommendation_made = False

        try:
            from core.embeddings import embed_text
            self._greet_vec = embed_text(
                "hi hello hey good morning good afternoon good evening greetings "
                "howdy what's up just saying hello nothing else no request"
            )
            self._booking_vec = embed_text(
                "book reschedule cancel appointment haircut massage facial spa "
                "manicure pedicure available slots today tomorrow what time"
            )
            print("DEBUG BRAIN: greeting/booking vectors pre-computed")
        except Exception:
            self._greet_vec  = None
            self._booking_vec = None

    @staticmethod
    def _is_resumable(data: dict) -> bool:
        """Never resume sessions across worker restarts.
        Slot-pick state is volatile (tied to a specific failed attempt + stale date/time/user).
        Resuming it causes the next user to see someone else's booking state immediately.
        Always start fresh -- the user will just re-state what they want.
        """
        return False

    def _save(self):
        save_session(self.session.__dict__)

    def handle_text(self, text: str) -> str:
        try:
            result = self._handle_text_inner(text)
            self._save()
            return result
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return f"An internal error occurred: {exc}"

    def _run_booking(self) -> str:
        """Execute booking and handle result."""

        _profile_name = self.profile.get("name", "")
        _profile_email = self.profile.get("email", "")
        
        if (_profile_name and _profile_email 
            and not getattr(self.session, "_contact_reuse_asked", False)
            and not self.session.name
            and not self.session.email):
            
            self.session._contact_reuse_asked = True  # Mark that we asked
            self.session._awaiting_contact_confirm = True
            self.session.awaiting_confirmation = False
            return (f"Should I use the same details as before? "
                    f"Name: {_profile_name}, Email: {_profile_email}?")

        result = run_booking_flow(
            session=self.session,
            name=self.session.name,
            email=self.session.email,
            phone=self.session.phone or "0000000000",
            date=self.session.date,
            time=self.session.time,
            timezone="Asia/Kolkata",
        )

        if result["type"] == "SLOT_UNAVAILABLE":
            slots = result.get("suggested_slots", [])
            self.session.suggested_slots = slots
            self.session.awaiting_confirmation = False
            self.session.awaiting_slot_pick = True
            if not slots:
                return "That time isn't available. Want to try a different date?"
            from datetime import datetime as _sudt
            def _su_fmt(t):
                try: return _sudt.strptime(t, "%H:%M").strftime("%I:%M %p").lstrip("0")
                except Exception: return t
            _su_labels = [_su_fmt(s) for s in slots]
            if len(_su_labels) == 1:
                return f"That time isn't available. How about {_su_labels[0]}?"
            elif len(_su_labels) == 2:
                return f"That time isn't available. How about {_su_labels[0]} or {_su_labels[1]}?"
            else:
                return (f"That time isn't available. How about "
                        f"{_su_labels[0]}, {_su_labels[1]}, or {_su_labels[2]}?")

        if result["type"] == "SAY":
            _say_txt = result["text"]
            # If "past time" error and time is AM hour -- offer PM equivalent automatically
            if "already passed" in _say_txt:
                try:
                    from core.time_utils import is_past_datetime as _ipd
                    _h, _m = map(int, self.session.time.split(":"))
                    if _h < 12:
                        _pm_time = f"{_h+12:02d}:{_m:02d}"
                        if not _ipd(self.session.date, _pm_time, "Asia/Kolkata"):
                            self.session.time = _pm_time
                            self.session.awaiting_confirmation = True
                            _svcs_pm = ", ".join(self.session.items_selected) if self.session.items_selected else "your appointment"
                            return (f"Did you mean {_pm_time}? So that would be {self.session.date} at {_pm_time} "
                                    f"for {_svcs_pm}. Shall I go ahead?")
                except Exception:
                    pass
            suggested_date, suggested_time = extract_datetime(_say_txt)
            if suggested_date:
                self.session.date = suggested_date
            if suggested_time:
                self.session.time = suggested_time
            self.session.awaiting_confirmation = True
            return _say_txt

        if result["type"] == "BOOKED":
            _bk_uid   = result.get("booking_uid")
            _bk_id    = self.session.booking_id
            _bk_name  = self.session.name
            _bk_email = self.session.email
            _bk_phone = self.session.phone
            _bk_date  = self.session.date
            _bk_time  = self.session.time
            save_profile(
                email=_bk_email,
                name=_bk_name,
                phone=_bk_phone,
                booking_id=_bk_id,
                booking_uid=_bk_uid,
                booking_date=_bk_date,
                booking_time=_bk_time,
                items_selected=list(self.session.items_selected),
            )
            self.profile = load_profile(email=_bk_email)
            print(f"DEBUG profile after booking: {self.profile}")
            clear_session()
            # Reset booking fields but KEEP name/email/phone in memory
            # so next booking in same conversation doesn't re-ask
            _remembered_name  = _bk_name
            _remembered_email = _bk_email
            _remembered_phone = _bk_phone
            _remembered_bk_id   = _bk_id
            _remembered_bk_uid  = _bk_uid
            self.session.reset()
            self.session.name  = _remembered_name
            self.session.email = _remembered_email
            self.session.phone = _remembered_phone
            self.session.booking_id  = _remembered_bk_id
            self.session.booking_uid = _remembered_bk_uid
            self.session.completed = True
            self.recommendation_made = False
            self.session._contact_reuse_asked = False
            return "Done -- your appointment is booked! You will get a confirmation email."

        return "Something went wrong. Want to try again?"

    def _run_reschedule(self) -> str:
        print(f"DEBUG _run_reschedule ENTRY:")
        print(f"  session.email = {self.session.email!r}")
        print(f"  _reschedule_email = {getattr(self.session, '_reschedule_email', 'NOT SET')!r}")
        print(f"  profile.email = {self.profile.get('email')!r}")
        print(f"  _reschedule_booking = {getattr(self.session, '_reschedule_booking', None)}")
        
        if not self.profile and self.session.email:
            self.profile = load_profile(self.session.email)
        print(f"DEBUG _run_reschedule: session.email={self.session.email} _reschedule_email={getattr(self.session, '_reschedule_email', None)} profile.email={self.profile.get('email')}")
        """Validate slot, cancel old booking, create new one."""
        _rb      = getattr(self.session, "_reschedule_booking", None)
        _old_uid = _rb.get("uid") if _rb else self.profile.get("booking_uid")
        _old_id  = _rb.get("id")  if _rb else self.profile.get("booking_id")
        _email   = (getattr(self.session, "_reschedule_email", None)
                        or self.session.email
                        or self.profile.get("email")
                        or "")
        _name = (self.session.name
                 or self.profile.get("name")
                 or (_rb.get("name") if _rb else None)
                 or (_rb.get("_att_name") if _rb else None)
                 or "Guest")
        # Guard against email parser bleeding into name field
        # (extract_user_details sometimes picks "Id" from email username)
        import re as _re_name
        if _name and _re_name.search(
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", _name
        ):
            _name = self.profile.get("name") or "Guest"

        _phone = self.session.phone or self.profile.get("phone", "0000000000")

        # Populate services from picked booking if session is empty
        if not self.session.items_selected and _rb:
            for s in _rb.get("services", []):
                self.session.add_item(s)

        # Safety: if email still empty, pull from profile one more time
        # Last resort: check reschedule_booking dict itself for embedded email
        if not _email and self.session._reschedule_booking:
            _email = self.session._reschedule_booking.get("email", "")
        
        if not _email:
            _email = (
                getattr(self.session, "_reschedule_email", None)
                or self.session.email
                or self.profile.get("email", "")
                or getattr(self.session, "email", "")
            )

        if not _email:
            self.session.awaiting_reschedule = True
            return "I need your email to complete the reschedule -- what email was the booking made with?"

        # Persist it so subsequent calls don't re-ask
        self.session._reschedule_email = _email
        self.session.email = _email

        # booking_flow now owns the full reschedule lifecycle:
        #   1. sanity checks (past time, weekend)
        #   2. slot availability check  ← protects old booking
        #   3. cancel old booking       ← only if new slot is open
        #   4. create new booking
        result = run_booking_flow(
            session=self.session,
            name=_name,
            email=_email,
            phone=_phone,
            date=self.session.date,
            time=self.session.time,
            timezone="Asia/Kolkata",
            is_reschedule=True,
            old_booking_id=_old_uid or _old_id,
        )

        if result["type"] == "SLOT_UNAVAILABLE":
            slots = result.get("suggested_slots", [])
            self.session.suggested_slots = slots
            self.session.awaiting_confirmation = False
            self.session.awaiting_reschedule_confirm = True
            self.session.awaiting_slot_pick = True if slots else False
            if not slots:
                self.session.awaiting_reschedule = True
                self.session.awaiting_reschedule_confirm = False
                return "That time isn't available. What other date or time works for you?"
            from datetime import datetime as _sudt
            def _su_fmt(t):
                try: return _sudt.strptime(t, "%H:%M").strftime("%I:%M %p").lstrip("0")
                except Exception: return t
            _su_labels = [_su_fmt(s) for s in slots]
            if len(_su_labels) == 1:
                return f"That time isn't available. How about {_su_labels[0]}?"
            elif len(_su_labels) == 2:
                return f"That time isn't available. How about {_su_labels[0]} or {_su_labels[1]}?"
            else:
                return (f"That time isn't available. How about "
                        f"{_su_labels[0]}, {_su_labels[1]}, or {_su_labels[2]}?")

        if result["type"] == "SAY":
            _say_txt = result["text"]
            if "weekend" in _say_txt.lower():
                self.session.awaiting_reschedule = False
                self.session.awaiting_reschedule_confirm = True
                self.session.awaiting_confirmation = True
            else:
                self.session.awaiting_reschedule = True
                self.session.awaiting_reschedule_confirm = False
                self.session.awaiting_confirmation = False
            return _say_txt

        if result["type"] == "BOOKED":
            _use_name  = _name
            _use_email = _email
            _new_date  = self.session.date
            _new_time  = self.session.time
            _svcs      = ", ".join(self.session.items_selected) if self.session.items_selected else "your appointment"
            save_profile(
                email=_use_email,
                name=_use_name,
                phone=_phone,
                booking_id=self.session.booking_id,
                booking_uid=result.get("booking_uid"),
                booking_date=_new_date,
                booking_time=_new_time,
                items_selected=list(self.session.items_selected),
            )
            self.profile = load_profile(email=_use_email)
            clear_session()
            _rem_name  = _use_name
            _rem_email = _use_email
            _rem_phone = _phone
            self.session.reset()
            self.session.name  = _rem_name
            self.session.email = _rem_email
            self.session.phone = _rem_phone
            self.session.completed = True
            self.recommendation_made = False
            self.session._contact_reuse_asked = False
            return (
                f"Done -- rescheduled to {_new_date} at {_new_time} for {_svcs}. "
                "You will get a confirmation email."
            )

        return "Something went wrong with the reschedule. Want to try again?"

    def _ensure_chosen_event(self):
        """Set session.chosen_event if not already set. Required before get_available_slots."""
        if not getattr(self.session, "chosen_event", None):
            from core.cal_metadata import fetch_events
            events = fetch_events()
            for e in events:
                if e.get("title", "").lower() == "salon appointment":
                    self.session.chosen_event = e
                    break

    def _list_bookings_for_pick(self, bookings: list, action: str) -> str:
        """
        Format a numbered list of bookings and ask user to pick one.
        action: "reschedule" or "cancel"
        """
        lines = [
            f"{i+1}. {_fmt_booking(b)}"
            for i, b in enumerate(bookings)
        ]
        return f"Which booking would you like to {action}?\n" + "\n".join(lines)

    def _handle_text_inner(self, text: str) -> str:
        log_event("user_message", {"text": text, "session": self.session.__dict__})

        if self.session.email and not self.profile:
            self.profile = load_profile(self.session.email)

        text_l = text.lower().strip()
        print(f"DEBUG STATE: awaiting_reschedule={self.session.awaiting_reschedule} "
                f"awaiting_reschedule_pick={getattr(self.session, 'awaiting_reschedule_pick', False)} "
                f"awaiting_reschedule_confirm={self.session.awaiting_reschedule_confirm} "
                f"awaiting_confirmation={self.session.awaiting_confirmation} "
                f"awaiting_slot_pick={self.session.awaiting_slot_pick}")

        import re as _re_top
        _has_email_top = bool(_re_top.search(
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text
        ))
        if _has_email_top and getattr(self.session, "_awaiting_contact_confirm", False):
            self.session._awaiting_contact_confirm = False
            self.session.awaiting_slot_pick = False
            self.session.awaiting_confirmation = False
            name, email, phone = extract_user_details(text)
            self.session.name  = name  if name  else None
            self.session.email = email if email else None
            self.session.phone = phone or "0000000000"
            if self.session.name and self.session.email:
                return self._run_booking()
            self.session.awaiting_user_details = True
            return "No problem -- what's your name and email address?"

        # GREETING
        _any_flow_active = (
            self.session.awaiting_confirmation
            or self.session.awaiting_slot_pick
            or self.session.awaiting_user_details
            or self.session.awaiting_reschedule
            or self.session.awaiting_reschedule_confirm
            or getattr(self.session, "awaiting_reschedule_pick", False)
            or self.session.awaiting_cancel_email
            or getattr(self.session, "awaiting_cancel_pick", False)
            or getattr(self.session, "awaiting_cancel_confirm", False)
            or self.session.awaiting_cancel_reason
            or self.session.awaiting_cancel_retry
            or getattr(self.session, "_awaiting_contact_confirm", False)
            or getattr(self.session, "_awaiting_reschedule_current_confirm", False)
            or getattr(self.session, "awaiting_weekend_choice", False)
            or getattr(self.session, "awaiting_rebook_after_cancel", False)
            or self.session.completed
            or bool(self.session.items_selected)
            or bool(self.session.date)
        )
        print(f"DEBUG ANY_FLOW: active={_any_flow_active} "
              f"conf={self.session.awaiting_confirmation} "
              f"slot={self.session.awaiting_slot_pick} "
              f"items={bool(self.session.items_selected)} "
              f"date={bool(self.session.date)}")

        if not _any_flow_active:
            try:
                from core.embeddings import embed_text
                from sentence_transformers import util
                if self._greet_vec is None or self._booking_vec is None:
                    raise ValueError("vectors not ready")
                _txt_vec = embed_text(text_l)
                _g_score = float(util.cos_sim(_txt_vec, self._greet_vec))
                _b_score = float(util.cos_sim(_txt_vec, self._booking_vec))
                print(f"DEBUG GREET: greet={_g_score:.3f} booking={_b_score:.3f}")
                _is_pure_greeting = _g_score > _b_score and _g_score >= 0.15
            except Exception:
                _is_pure_greeting = text_l in {"hi", "hello", "hey", "hiya",
                                                "howdy", "good morning", "good afternoon",
                                                "good evening", "greetings", "sup", "what's up"}

            if _is_pure_greeting:
                return ("Hi there! Welcome to the salon. "
                        "I can help you book, reschedule, or cancel an appointment. "
                        "What would you like to do today?")

        # EXIT
        if text_l in {"exit", "quit", "stop session", "end call"}:
            return "__END_SESSION__"

        # LIGHT RESET
        if is_reject(text) and text_l in {"never mind", "start over", "reset"}:
            self.session.reset()
            self.recommendation_made = False
            clear_session()
            return "Done, all cleared. What would you like to book?"

        # ======================================================================
        # MID-CONVERSATION RESCHEDULE INTENT
        # User asks to reschedule while a booking is active in the current session
        # (session.completed is False but session has booking data)
        # ======================================================================
        # Words that indicate a slot inquiry, not a reschedule request
        _slot_inquiry_guard = any(w in text_l for w in {
            "available", "slot", "time slot", "which time", "what time",
            "open", "free", "earliest", "soonest", "when can"
        })

        _mid_resch = (
            not self.session.completed
            and not self.session.awaiting_reschedule
            and not self.session.awaiting_reschedule_email
            and not self.session.awaiting_reschedule_pick
            and not self.session.awaiting_reschedule_confirm
            and not self.session.awaiting_cancel_retry
            and not getattr(self.session, "_awaiting_reschedule_current_confirm", False)
            and not self.session.awaiting_user_details
            and not getattr(self.session, "awaiting_weekend_choice", False)
            and not _slot_inquiry_guard
            and (
                is_reschedule_intent(text)
                or "reschedule" in text_l
                or "move it to" in text_l
                or ("change it to" in text_l and not self.session.awaiting_confirmation)
                or "change my appointment" in text_l
                or "move my appointment" in text_l
            )
        )

        if _mid_resch:
            # Case A: active booking in this session -- ask if it's this one
            _has_active_booking = bool(
                self.session.items_selected
                and self.session.date
                and self.session.time
            )
            if _has_active_booking:
                _svc_mid  = ", ".join(self.session.items_selected)
                _date_mid = self.session.date
                _time_mid = self.session.time
                # Store current booking info so the confirm handler can use it
                self.session._pending_reschedule_id  = self.session.booking_id
                self.session._pending_reschedule_uid = self.session.booking_uid
                self.session._awaiting_reschedule_current_confirm = True
                self.session._reschedule_email = (
                self.session.email
                or self.profile.get("email", "")
                )
                self.session._awaiting_reschedule_current_confirm = True
                return (
                    f"Would you like to reschedule your {_svc_mid} "
                    f"on {_date_mid} at {_time_mid}?"
                )

            # Case B: no active booking in session -- go straight to email
            self.session.awaiting_reschedule = True
            self.session.awaiting_reschedule_email = True
            return "What email was the booking made with?"

        # ======================================================================
        # MID-CONVERSATION CANCEL INTENT
        # User asks to cancel while a booking is active in the current session
        # ======================================================================
        _mid_cancel = (
            not self.session.completed
            and not self.session.awaiting_cancel_email
            and not getattr(self.session, "awaiting_cancel_pick", False)
            and not getattr(self.session, "awaiting_cancel_confirm", False)
            and not self.session.awaiting_cancel_reason
            and not self.session.awaiting_cancel_retry
            and not getattr(self.session, "_awaiting_cancel_current_confirm", False)
            and not self.session.awaiting_confirmation   
            and not self.session.awaiting_user_details
            and not getattr(self.session, "awaiting_weekend_choice", False)
            and not _slot_inquiry_guard
            and (
                is_cancel_intent(text)
                or has_cancel_keyword(text)
            )
        )

        if _mid_cancel:
            # Case A: active booking in this session -- ask if it's this one
            _has_active_booking_c = bool(
                self.session.items_selected
                and self.session.date
                and self.session.time
            )
            if _has_active_booking_c:
                _svc_c  = ", ".join(self.session.items_selected)
                _date_c = self.session.date
                _time_c = self.session.time
                self.session._pending_cancel_id  = self.session.booking_id
                self.session._pending_cancel_uid = self.session.booking_uid
                self.session._awaiting_cancel_current_confirm = True
                return (
                    f"Would you like to cancel your {_svc_c} "
                    f"on {_date_c} at {_time_c}?"
                )

            # Case B: no active booking -- go straight to email
            self.session.awaiting_cancel_email = True
            return "What email was the booking made with?"

        # ======================================================================
        # POST-COMPLETION HANDLER
        # When a booking/reschedule just finished and user asks to book/reschedule again.
        # ======================================================================
        if self.session.completed:
            # Hard keyword guard -- never let "reschedule" be overridden by _early_bk
            _is_resch_post = (
                is_reschedule_intent(text)
                or "reschedule" in text_l
                or "rescheduling" in text_l
                or "move it to" in text_l
                or "change it to" in text_l
            )
            # --- RESCHEDULE after completion: use profile booking data, no session reset ---
            if _is_resch_post:
                if not self.profile and self.session.email:
                    self.profile = load_profile(self.session.email)
                _p_email = self.profile.get("email") or self.session.email
                _p_name  = self.profile.get("name") or self.session.name
                # Restore session state
                self.session.completed = False
                self.session.awaiting_reschedule = True
                self.session.awaiting_reschedule_email = False
                self.session.name  = _p_name or self.session.name or ""
                self.session.email = _p_email or self.session.email or ""
                self.session._reschedule_email = _p_email or self.session.email or ""
                # Look up ALL bookings by email -- user may have multiple
                if _p_email:
                    from bookings import lookup_booking_by_email as _lbe_post
                    _post_bookings = _lbe_post(_p_email)
                    if _post_bookings:
                        if len(_post_bookings) > 1:
                            # If we know the just-booked UID, use it directly
                            _just_booked_uid = self.session.booking_uid
                            if _just_booked_uid:
                                _matched = next(
                                    (b for b in _post_bookings if b.get("uid") == _just_booked_uid),
                                    None
                                )
                                if _matched:
                                    self.session._reschedule_booking = _matched
                                    _p_svcs = list(_matched.get("services", []))
                                    for _sv in _p_svcs:
                                        self.session.add_item(_sv)
                                    # If message has target time, jump to confirm
                                    _fd, _ = extract_datetime(text)
                                    # Multi-time scan: pick first time that differs from booking's orig time
                                    import re as _re_post
                                    import pytz as _ptz_post
                                    from datetime import datetime as _dt_post
                                    # Parse orig time from matched booking's ISO start_time
                                    _orig_bk_time = None
                                    try:
                                        _st_m = _re_post.sub(r"\.\d+Z$", "Z", _matched.get("start_time", "")).replace(".000Z","Z").replace("Z","+00:00")
                                        _orig_bk_time = _dt_post.fromisoformat(_st_m).astimezone(_ptz_post.timezone("Asia/Kolkata")).strftime("%H:%M")
                                    except Exception:
                                        _orig_bk_time = None

                                    _all_times_post = _re_post.findall(r"\b(1[0-2]|0?[1-9])(?::(\d{2}))?\s*(am|pm|AM|PM)\b", text)
                                    _ft = None
                                    for _tm in _all_times_post:
                                        try:
                                            _hh = int(_tm[0]); _mm = int(_tm[1]) if _tm[1] else 0; _ap = _tm[2].lower()
                                            if _ap == "pm" and _hh != 12: _hh += 12
                                            if _ap == "am" and _hh == 12: _hh = 0
                                            _cand = f"{_hh:02d}:{_mm:02d}"
                                            if _cand != _orig_bk_time:
                                                _ft = _cand
                                                break
                                        except Exception:
                                            pass
                                    if not _ft:
                                        _, _ft = extract_datetime(text)  # fallback
                                    if _ft:
                                        if _fd:
                                            self.session.date = _fd
                                        elif not self.session.date:
                                            # Fall back to the matched booking's own date
                                            try:
                                                import pytz as _ptz_fd
                                                from datetime import datetime as _dt_fd
                                                _st_fd = _re_post.sub(r"\.\d+Z$", "Z", _matched.get("start_time","")).replace(".000Z","Z").replace("Z","+00:00")
                                                self.session.date = _dt_fd.fromisoformat(_st_fd).astimezone(_ptz_fd.timezone("Asia/Kolkata")).strftime("%Y-%m-%d")
                                            except Exception:
                                                pass
                                        self.session.time = _ft
                                        _svcs_fc = ", ".join(self.session.items_selected) or "your appointment"
                                        self.session.awaiting_reschedule = False
                                        self.session.awaiting_reschedule_confirm = True
                                        self.session.awaiting_confirmation = True
                                        return f"Got it -- reschedule to {self.session.date} at {_ft} for {_svcs_fc}. Shall I go ahead?"
                                    return f"Found it -- {_fmt_booking(_matched)}. What date and time would you like to move it to?"
                                
                            # No match or no UID -- show list
                            self.session._reschedule_options = _post_bookings
                            self.session.awaiting_reschedule_pick = True
                            return self._list_bookings_for_pick(_post_bookings, "reschedule")
                        else:
                            self.session._reschedule_booking = _post_bookings[0]
                            _p_svcs = list(
                                self.profile.get("items_selected")
                                or _post_bookings[0].get("services", [])
                            )
                            for _sv in _p_svcs:
                                self.session.add_item(_sv)
                            return (
                                f"Found it -- {_fmt_booking(_post_bookings[0])}. "
                                "What date and time would you like to move it to?"
                            )

                # Fallback: use profile data (single stored booking)
                _p_uid  = self.profile.get("booking_uid")
                _p_id   = self.profile.get("booking_id")
                _p_date = self.profile.get("booking_date")
                _p_time = self.profile.get("booking_time")
                _p_svcs = list(self.profile.get("items_selected") or [])
                self.session._reschedule_booking = {
                    "id":         _p_id,
                    "uid":        _p_uid,
                    "start_time": f"{_p_date}T{_p_time.replace(':','')[:4]}:00+05:30" if _p_date and _p_time else "",
                    "services":   _p_svcs,
                }
                for _sv in _p_svcs:
                    self.session.add_item(_sv)
                # If message already has the target date/time, jump to confirm
                _fd, _ = extract_datetime(text)
                # Multi-time scan: pick first time that differs from booking's orig time
                import re as _re_post
                _orig_bk_time = _p_time

                _all_times_post = _re_post.findall(r"\b(1[0-2]|0?[1-9])(?::(\d{2}))?\s*(am|pm|AM|PM)\b", text)
                _ft = None
                for _tm in _all_times_post:
                    try:
                        _hh = int(_tm[0]); _mm = int(_tm[1]) if _tm[1] else 0; _ap = _tm[2].lower()
                        if _ap == "pm" and _hh != 12: _hh += 12
                        if _ap == "am" and _hh == 12: _hh = 0
                        _cand = f"{_hh:02d}:{_mm:02d}"
                        if _cand != _orig_bk_time:
                            _ft = _cand
                            break
                    except Exception:
                        pass
                if not _ft:
                    _, _ft = extract_datetime(text)  # fallback

                if _ft:
                    if _fd:
                        self.session.date = _fd
                    elif _p_date:
                        self.session.date = _p_date
                    self.session.time = _ft
                    _svcs_fc = ", ".join(self.session.items_selected) if self.session.items_selected else "your appointment"
                    _cur_fc = f" (currently {_p_date} at {_p_time})" if _p_date and _p_time else ""
                    self.session.awaiting_reschedule = False
                    self.session.awaiting_reschedule_confirm = True
                    self.session.awaiting_confirmation = True
                    return f"Got it -- reschedule{_cur_fc} to {self.session.date} at {_ft} for {_svcs_fc}. Shall I go ahead?"
                _prev = ""
                if _p_date and _p_time:
                    _prev = f" (currently {_p_date} at {_p_time})"
                return f"Sure{_prev}. What date and time would you like to move it to?"

            # Never treat cancel pick/confirm messages as a new booking trigger
            _any_cancel_active = (
                self.session.awaiting_cancel_email
                or getattr(self.session, "awaiting_cancel_pick", False)
                or getattr(self.session, "awaiting_cancel_confirm", False)
                or self.session.awaiting_cancel_reason
                or self.session.awaiting_cancel_retry
            )
            # Use word-boundary check so "booking" doesn't match "book"
            _early_bk = not _any_cancel_active and (
                " book" in (" " + text_l) or "appointment" in text_l or
                "schedule" in text_l or "haircut" in text_l or
                "facial" in text_l or "massage" in text_l or
                "manicure" in text_l or "pedicure" in text_l or
                "hair spa" in text_l)
            if _early_bk:
                # Extract service/date/time from THIS message before clearing
                # so "book facial Friday at 3" retains context after identity question
                from core.extractor import extract_service as _exsvc
                _pre_svc = _exsvc(text, CATALOG)
                _pre_date, _pre_time = extract_datetime(text)
                # Reset booking fields so old state doesn't leak
                self.session.items_selected = []
                self.session.date = None
                self.session.time = None
                self.session.awaiting_confirmation = False
                self.session.awaiting_slot_pick = False
                self.session.suggested_slots = []
                self.session.completed = False
                self.recommendation_made = False
                # Re-apply service/date/time from triggering message
                if _pre_svc:
                    self.session.domain = _pre_svc[0]
                    self.session.add_item(_pre_svc[1])
                if _pre_date:
                    self.session.date = _pre_date
                if _pre_time:
                    self.session.time = _pre_time
                _pname  = self.profile.get("name")
                _pemail = self.profile.get("email")
                if _pname and _pemail:
                    self.session._awaiting_same_identity = True
                    return (f"Sure! Would you like to book with the same name ({_pname}) "
                            f"and email ({_pemail}) as before? Or a different one?")
                # No profile -- fall through to normal booking flow

            # Graceful stop -- user is done, don't fall into booking flow
            if self.session.completed:
                _done_words = {
                    "no", "nope", "nothing", "that's all", "thats all",
                    "all good", "i'm good", "im good", "no thanks",
                    "thank you", "thanks", "bye", "goodbye", "that's it",
                    "thats it", "i'm done", "im done", "all set", "great",
                    "perfect", "awesome", "cool", "okay", "ok"
                }
                if is_reject(text) or any(w in text_l for w in _done_words):
                    self.session.completed = False
                    return "No problem! Let me know if you need anything else."

        # SAME-IDENTITY CONFIRM
        if getattr(self.session, "_awaiting_same_identity", False):
            self.session._awaiting_same_identity = False
            _pname  = self.profile.get("name", "")
            _pemail = self.profile.get("email", "")
            if is_confirm(text) or any(w in text_l for w in {"same", "yes", "yeah", "yep", "sure", "ok", "okay", "keep"}):
                # Reuse profile identity
                self.session.name  = _pname
                self.session.email = _pemail
                self.session.phone = self.profile.get("phone", "0000000000")

                # Check if we already have all booking info from the triggering message
                _has_service = bool(self.session.items_selected)
                _has_date = bool(self.session.date)
                _has_time = bool(self.session.time)

                if _has_service and _has_date and _has_time:
                    # All info present - go straight to confirmation
                    self.session.awaiting_confirmation = True
                    svcs = ", ".join(self.session.items_selected)
                    return f"Got it! {self.session.date} at {self.session.time} for {svcs}. Should I book that?"
                elif _has_service and _has_date:
                    return f"Great! What time on {self.session.date} works for you?"
                elif _has_service:
                    svcs = ", ".join(self.session.items_selected)
                    return f"Got it! {svcs}. What date and time works for you?"
                else:
                    return "Great! What service would you like? Haircut, Hair Spa, Massage, Facial, Manicure, or Pedicure."

            else:
                # Fresh identity -- collect name+email now
                self.session.name  = None
                self.session.email = None
                # If service/date/time already known, go straight to name+email
                if self.session.items_selected and self.session.date and self.session.time:
                    self.session.awaiting_user_details = True
                    _svcs_id = ", ".join(self.session.items_selected)
                    return f"Got it! What's the name and email for the {_svcs_id} booking?"
                elif self.session.items_selected:
                    return "Got it! What date and time works for you?"
                return "Got it! What service would you like? Haircut, Hair Spa, Massage, Facial, Manicure, or Pedicure."

        # CONTACT REUSE CONFIRM
        # User is responding to "Should I use same name/email as before?"
        if getattr(self.session, "_awaiting_contact_confirm", False):
            self.session._awaiting_contact_confirm = False
            if is_confirm(text) or any(w in text_l for w in
                                    {"same", "yes", "yeah", "yep", "sure", "ok",
                                        "okay", "keep", "use", "correct", "right"}):
                # Reuse — go straight to booking
                return self._run_booking()
            else:
                # They want different details -- clear stale identity FIRST
                # so old profile name/email cannot bleed through if extraction fails
                self.session.name = None
                self.session.email = None
                name, email, phone = extract_user_details(text)
                if name:
                    self.session.name = name
                if email:
                    self.session.email = email
                if phone:
                    self.session.phone = phone
                # If they gave both inline, book immediately
                if self.session.name and self.session.email:
                    return self._run_booking()
                # Otherwise collect fresh
                self.session.awaiting_user_details = True
                return "No problem -- what's your name and email address?"

        # RESCHEDULE CURRENT BOOKING CONFIRM
        # User responds to "Would you like to reschedule your [service] on [date]?"
        if getattr(self.session, "_awaiting_reschedule_current_confirm", False):
            self.session._awaiting_reschedule_current_confirm = False

            if is_confirm(text) or any(w in text_l for w in
                {"yes", "yeah", "yep", "sure", "ok", "okay", "that one",
                "this one", "correct", "right", "yup", "absolutely"}):
                self.session.awaiting_reschedule = True
                self.session.awaiting_reschedule_email = False
                _cur_svcs = ", ".join(self.session.items_selected) if self.session.items_selected else "your appointment"
                # Use session data first -- it is always current during mid-conversation
                _cur_date = self.session.date or self.profile.get("booking_date") or ""
                _cur_time = self.session.time or self.profile.get("booking_time") or ""
                # Store as reschedule booking so _run_reschedule can find IDs
                self.session._reschedule_booking = {
                    "id":         self.session._pending_reschedule_id,
                    "uid":        self.session._pending_reschedule_uid,
                    "start_time": f"{_cur_date}T{_cur_time.replace(':','')}:00+05:30" if _cur_date and _cur_time else "",
                    "services":   list(self.session.items_selected),
                }
                self.session._reschedule_email = self.session.email or self.profile.get("email", "")
                # Also store on session.email so _run_reschedule fallback works
                if not self.session.email:
                    self.session.email = self.session._reschedule_email
                _prev = f" (currently {_cur_date} at {_cur_time})" if _cur_date and _cur_time else ""
                return f"Sure{_prev}. What date and time would you like to move it to?"

            else:
                # User wants to reschedule a different booking -- ask if same email
                self.session._awaiting_reschedule_same_email = True
                _pe = self.profile.get("email", "")
                if _pe:
                    return (f"No problem. Is the other booking also under {_pe}? "
                            f"Or was it a different email?")
                # No profile email -- ask directly
                self.session.awaiting_reschedule = True
                self.session.awaiting_reschedule_email = True
                return "What email was that booking made with?"

        # RESCHEDULE SAME EMAIL CONFIRM
        # User responds to "Is it under [email]?"
        if getattr(self.session, "_awaiting_reschedule_same_email", False):
            self.session._awaiting_reschedule_same_email = False
            _pe = self.profile.get("email", "")

            import re as _re_rsame
            _typed_email = _re_rsame.search(
                r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text
            )

            if _typed_email:
                # User typed a different email directly
                _use_email = _typed_email.group(0).strip()
            elif is_confirm(text) or any(w in text_l for w in
                    {"yes", "yeah", "same", "that", "correct", "right",
                     "yep", "sure", "same email", "that email"}):
                _use_email = _pe
            else:
                # User said no -- ask for email
                self.session.awaiting_reschedule = True
                self.session.awaiting_reschedule_email = True
                return "What email was that booking made with?"

            # Look up bookings under chosen email
            from bookings import lookup_booking_by_email
            _bookings = lookup_booking_by_email(_use_email)
            if not _bookings:
                self.session.awaiting_reschedule = True
                self.session.awaiting_reschedule_email = True
                return (f"I couldn't find any bookings under {_use_email}. "
                        "Could you double-check the email?")

            self.session._reschedule_email = _use_email
            self.session.awaiting_reschedule = True
            self.session.awaiting_reschedule_email = False

            if len(_bookings) == 1:
                self.session._reschedule_booking = _bookings[0]
                return f"Found it -- {_fmt_booking(_bookings[0])}. What date and time would you like to move it to?"
            else:
                self.session._reschedule_options = _bookings
                self.session.awaiting_reschedule_pick = True
                return self._list_bookings_for_pick(_bookings, "reschedule")

        # CANCEL CURRENT BOOKING CONFIRM
        # User responds to "Would you like to cancel your [service] on [date]?"
        if getattr(self.session, "_awaiting_cancel_current_confirm", False):
            self.session._awaiting_cancel_current_confirm = False

            if is_confirm(text) or any(w in text_l for w in
                    {"yes", "yeah", "yep", "sure", "ok", "okay", "that one",
                     "this one", "correct", "right", "yup", "absolutely"}):
                # Pull IDs from pending flags first, fall back to session then profile
                self.session._found_booking_id  = (
                    self.session._pending_cancel_id
                    or self.session.booking_id
                    or self.profile.get("booking_id")
                )
                self.session._found_booking_uid = (
                    self.session._pending_cancel_uid
                    or self.session.booking_uid
                    or self.profile.get("booking_uid")
                )
                # Store email so cancel handler knows whose booking this is
                self.session._last_cancel_email = (
                    self.session.email
                    or self.profile.get("email", "")
                )
                self.session.awaiting_cancel_reason = True
                return ("Sure, cancelling that. Would you mind sharing the reason? "
                        "It's completely optional -- just helps the salon.")

            else:
                # Different booking -- ask if same email
                self.session._awaiting_cancel_same_email = True
                _pe = self.profile.get("email", "")
                if _pe:
                    return (f"No problem. Is the other booking also under {_pe}? "
                            f"Or was it a different email?")
                self.session.awaiting_cancel_email = True
                return "What email was that booking made with?"

        # CANCEL SAME EMAIL CONFIRM
        # User responds to "Is it under [email]?"
        if getattr(self.session, "_awaiting_cancel_same_email", False):
            self.session._awaiting_cancel_same_email = False
            _pe = self.profile.get("email", "")

            import re as _re_csame
            _typed_email_c = _re_csame.search(
                r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text
            )

            if _typed_email_c:
                _use_email_c = _typed_email_c.group(0).strip()
            elif is_confirm(text) or any(w in text_l for w in
                    {"yes", "yeah", "same", "that", "correct", "right",
                     "yep", "sure", "same email", "that email"}):
                _use_email_c = _pe
            else:
                self.session.awaiting_cancel_email = True
                return "What email was that booking made with?"

            from bookings import lookup_booking_by_email
            _bookings_c = lookup_booking_by_email(_use_email_c)
            if not _bookings_c:
                self.session.awaiting_cancel_email = True
                return (f"I couldn't find any bookings under {_use_email_c}. "
                        "Could you double-check the email?")

            self.session._last_cancel_email  = _use_email_c
            self.session._all_looked_up_bookings = _bookings_c

            if len(_bookings_c) == 1:
                self.session._pending_cancel_id  = _bookings_c[0]["id"]
                self.session._pending_cancel_uid = _bookings_c[0]["uid"]
                self.session._found_booking_id   = _bookings_c[0]["id"]
                self.session._found_booking_uid  = _bookings_c[0]["uid"]
                self.session.awaiting_cancel_reason = True
                return (f"Found it -- {_fmt_booking(_bookings_c[0])}. "
                        "Would you mind sharing the reason for cancelling? "
                        "Completely optional.")
            else:
                self.session._cancel_booking_options = _bookings_c
                self.session.awaiting_cancel_pick = True
                return self._list_bookings_for_pick(_bookings_c, "cancel")

        # STALE STATE GUARD
        # Only clear slot-pick state for fresh bookings, not reschedule flows
        _date, _time = extract_datetime(text)
        if (_date or _time) and self.session.awaiting_slot_pick and not self.session.awaiting_reschedule_confirm:
            self.session.awaiting_slot_pick = False
            self.session.awaiting_confirmation = False
            self.session.suggested_slots = []
            if _date and _time:
            # Only overwrite date if BOTH date and time are explicit in the message
            # Prevents "the one at 7pm" from injecting a phantom date
                self.session.date = _date
            if _time:
                self.session.time = _time

        # ======================================================================
        # PRIORITY BLOCK
        # ======================================================================
        # ========== WEEKEND CHOICE HANDLER (FULLY SEMANTIC - NO HARDCODED PATTERNS) ==========
        if getattr(self.session, "awaiting_weekend_choice", False):

            next_wd          = self.session._weekend_next_weekday
            requested_time   = self.session._weekend_requested_time
            available_times  = getattr(self.session, "_weekend_all_slots", [])
            same_time_avail  = requested_time in available_times

            def _fmt_time(t):
                try:
                    from datetime import datetime as _dt_wc2
                    return _dt_wc2.strptime(t, "%H:%M").strftime("%I:%M %p").lstrip("0")
                except Exception:
                    return t

            def _t2m(t):
                h, m = map(int, t.split(":")); return h * 60 + m

            # ── Use embeddings to detect intent — no hardcoded phrases ────────
            from core.embeddings import embed_text
            from sentence_transformers import util

            if same_time_avail:
                # User was asked: "Would you like the same time (3PM) on Monday?"
                yes_desc = (
                    "yes confirm agree same time that time works book it go ahead "
                    "sounds good that works perfect okay sure yep absolutely"
                )
                no_desc = (
                    "no different time other time change earliest slots available "
                    "show me options what else is available no thank you"
                )
                try:
                    u_vec   = embed_text(text)
                    y_score = float(util.cos_sim(u_vec, embed_text(yes_desc)))
                    n_score = float(util.cos_sim(u_vec, embed_text(no_desc)))
                    print(f"DEBUG WEEKEND YES={y_score:.3f} NO={n_score:.3f}")
                    _wants_same = y_score > n_score and y_score >= 0.15
                    _wants_diff = n_score > y_score and n_score >= 0.15
                except Exception:
                    _wants_same = is_confirm(text)
                    _wants_diff = is_reject(text)

                if _wants_same:
                    # Book same time on next weekday
                    self.session.awaiting_weekend_choice = False
                    self.session.__dict__.pop('_weekend_original_date', None)
                    self.session.__dict__.pop('_weekend_next_weekday', None)
                    self.session.__dict__.pop('_weekend_requested_time', None)
                    self.session.__dict__.pop('_weekend_all_slots', None)
                    self.session.date = next_wd
                    self.session.time = requested_time
                    self.session.awaiting_confirmation = True
                    if not self.session.name or not self.session.email:
                        self.session.awaiting_user_details = True
                        self.session.awaiting_confirmation = False
                        return "Almost there -- what's your name and email address?"
                    return self._run_booking()

                elif _wants_diff:
                    # User doesn't want same time — show 3 earliest
                    self.session.awaiting_weekend_choice = False
                    self.session.date = next_wd
                    earliest_3 = sorted(available_times)[:3]
                    self.session.suggested_slots = earliest_3
                    self.session.awaiting_slot_pick = True
                    self.session.__dict__.pop('_weekend_original_date', None)
                    self.session.__dict__.pop('_weekend_next_weekday', None)
                    self.session.__dict__.pop('_weekend_requested_time', None)
                    self.session.__dict__.pop('_weekend_all_slots', None)
                    if not earliest_3:
                        return f"{next_wd} has no available slots. Want to try a different date?"
                    _labels = [_fmt_time(t) for t in earliest_3]
                    if len(_labels) == 1:
                        return f"The earliest available time on {next_wd} is {_labels[0]}. Does that work?"
                    elif len(_labels) == 2:
                        return f"The earliest available times on {next_wd} are {_labels[0]} or {_labels[1]}. Which works?"
                    else:
                        return (
                            f"The earliest available times on {next_wd} are "
                            f"{_labels[0]}, {_labels[1]}, or {_labels[2]}. Which works?"
                        )
                else:
                    # Ambiguous — ask again
                    return (
                        f"Would you like the same time ({_fmt_time(requested_time)}) "
                        f"on {next_wd}?"
                    )

            else:
                # Same time NOT available — user was asked: "Nearest to 3PM or earliest?"
                nearest_desc = (
                    "nearest closest similar around that time near that slot "
                    "close to that time nearest available around the same time"
                )
                earliest_desc = (
                    "earliest soonest first available morning first slot "
                    "earliest of the day very first as early as possible"
                )
                try:
                    u_vec   = embed_text(text)
                    n_score = float(util.cos_sim(u_vec, embed_text(nearest_desc)))
                    e_score = float(util.cos_sim(u_vec, embed_text(earliest_desc)))
                    print(f"DEBUG WEEKEND NEAREST={n_score:.3f} EARLIEST={e_score:.3f}")
                    _wants_nearest  = n_score > e_score and n_score >= 0.15
                    _wants_earliest = e_score > n_score and e_score >= 0.15
                except Exception:
                    _wants_nearest  = False
                    _wants_earliest = False

                self.session.awaiting_weekend_choice = False
                self.session.date = next_wd
                self.session.__dict__.pop('_weekend_original_date', None)
                self.session.__dict__.pop('_weekend_next_weekday', None)
                self.session.__dict__.pop('_weekend_requested_time', None)
                self.session.__dict__.pop('_weekend_all_slots', None)

                if _wants_nearest:
                    _req_m   = _t2m(requested_time)
                    nearest_3 = sorted(
                        available_times,
                        key=lambda t: abs(_t2m(t) - _req_m)
                    )[:3]
                    nearest_3 = sorted(nearest_3)
                    self.session.suggested_slots = nearest_3
                    self.session.awaiting_slot_pick = True
                    _labels = [_fmt_time(t) for t in nearest_3]
                    if len(_labels) == 1:
                        return f"The nearest available time is {_labels[0]}. Does that work?"
                    elif len(_labels) == 2:
                        return f"The nearest available times are {_labels[0]} or {_labels[1]}. Which works?"
                    else:
                        return (
                            f"The nearest available times to {_fmt_time(requested_time)} are "
                            f"{_labels[0]}, {_labels[1]}, or {_labels[2]}. Which works?"
                        )

                elif _wants_earliest:
                    earliest_3 = sorted(available_times)[:3]
                    self.session.suggested_slots = earliest_3
                    self.session.awaiting_slot_pick = True
                    _labels = [_fmt_time(t) for t in earliest_3]
                    if len(_labels) == 1:
                        return f"The earliest available time on {next_wd} is {_labels[0]}. Does that work?"
                    elif len(_labels) == 2:
                        return f"The earliest available times are {_labels[0]} or {_labels[1]}. Which works?"
                    else:
                        return (
                            f"The earliest available times on {next_wd} are "
                            f"{_labels[0]}, {_labels[1]}, or {_labels[2]}. Which works?"
                        )

                else:
                    # Ambiguous — ask again clearly
                    self.session.awaiting_weekend_choice = True  # keep in state
                    self.session._weekend_next_weekday   = next_wd
                    self.session._weekend_requested_time = requested_time
                    self.session._weekend_all_slots      = available_times
                    return (
                        f"Would you like the nearest slot to {_fmt_time(requested_time)}, "
                        f"or see the earliest available times on {next_wd}?"
                    )
            
        # ========== END WEEKEND CHOICE HANDLER ==========

        # SLOT PICK (shared by booking + reschedule)
        if self.session.awaiting_slot_pick:
            import re as _re_sp
            _has_email_sp = bool(_re_sp.search(
                r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text
            ))
            if _has_email_sp and getattr(self.session, "_awaiting_contact_confirm", False):
                self.session.awaiting_slot_pick = False
                self.session._awaiting_contact_confirm = False
                name, email, phone = extract_user_details(text)
                self.session.name  = name  if name  else None
                self.session.email = email if email else None
                self.session.phone = phone or "0000000000"
                if self.session.name and self.session.email:
                    return self._run_booking()
                self.session.awaiting_user_details = True
                return "No problem -- what's your name and email address?"

            import re as _slre
            from datetime import datetime as _sdt
            slots = self.session.suggested_slots or []
            picked = None
            _text_sl = text.lower().strip()

            def _slot_label(t):
                try: return _sdt.strptime(t, "%H:%M").strftime("%I:%M %p").lstrip("0")
                except Exception: return t
            slot_labels = [_slot_label(s) for s in slots]

            # STEP 1: parse_slot_choice handles ordinals (__FIRST__ etc.) + time strings
            _pc = parse_slot_choice(text)
            if _pc == "__FIRST__" and slots:
                picked = slots[0]
            elif _pc == "__SECOND__" and len(slots) >= 2:
                picked = slots[1]
            elif _pc == "__THIRD__" and len(slots) >= 3:
                picked = slots[2]
            elif _pc and _pc in slots:
                picked = _pc

            # STEP 2: Explicit am/pm -- "4pm", "6 am", "10:30pm"
            if not picked:
                _sl_m = _slre.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", _text_sl)
                if _sl_m:
                    _hh = int(_sl_m.group(1)); _mm = int(_sl_m.group(2) or 0); _ap = _sl_m.group(3)
                    if _ap == "pm" and _hh != 12: _hh += 12
                    elif _ap == "am" and _hh == 12: _hh = 0
                    _cand = f"{_hh:02d}:{_mm:02d}"
                    if _cand in slots:
                        picked = _cand

            # STEP 3: HH:MM colon format -- "15:00", "9:30"
            if not picked:
                _sl_m2 = _slre.search(r"\b(\d{1,2}):(\d{2})\b", text)
                if _sl_m2:
                    _cand2 = f"{int(_sl_m2.group(1)):02d}:{_sl_m2.group(2)}"
                    if _cand2 in slots:
                        picked = _cand2

            # STEP 4: Bare number -- 1-based index first, then as hour with PM bias
            # "3" with slots [15:00,16:00,17:00]: index->17:00 OR time->15:00
            if not picked:
                _bare = _slre.search(r"(?<![\d:])(\d{1,2})(?![\d:])", _text_sl)
                if _bare:
                    _n = int(_bare.group(1))
                    _index_match = slots[_n - 1] if 1 <= _n <= len(slots) else None
                    _time_cands = [f"{_n:02d}:00"]
                    if 1 <= _n <= 9:
                        _time_cands.append(f"{_n + 12:02d}:00")
                    _time_match = next((c for c in _time_cands if c in slots), None)
                    picked = _index_match or _time_match

            # STEP 5: Semantic similarity -- handles any natural phrasing without hardcoding
            # "the earliest one", "that late slot", "the middle one", "last available", etc.
            # Automatically understands forward AND reverse ordinals, relative positions
            if not picked:
                try:
                    from core.embeddings import embed_text
                    from sentence_transformers import util as _st_util
                    _descs = []
                    total_slots = len(slots)

                    for i, (s, lbl) in enumerate(zip(slots, slot_labels)):
                        forward_position = i
                        reverse_position = total_slots - 1 - i

                        # Build rich semantic description of this slot's position
                        position_tags = []

                        # Forward ordinal position
                        forward_labels = [
                            "first, earliest, top of list, first option, number one",
                            "second, next one, second option, number two",
                            "third, third option, number three",
                            "fourth, fourth option, number four",
                            "fifth, fifth option, number five",
                        ]
                        position_tags.append(forward_labels[i] if i < 5 else f"option number {i+1}")

                        # Reverse ordinal position - this was the critical missing piece!
                        reverse_labels = [
                            "last, final one, latest, end of list, bottom of list, last option",
                            "second last, second to last, penultimate, one before last, second from end",
                            "third last, third to last, third from end, two before last",
                            "fourth last, fourth to last, fourth from end",
                        ]
                        position_tags.append(reverse_labels[reverse_position] if reverse_position < 4 else f"{reverse_position+1}th from end")

                        # Extra relative position context
                        if forward_position == 0:
                            position_tags.append("earliest time, first available, soonest")
                        elif reverse_position == 0:
                            position_tags.append("latest time, last available, latest available")
                        elif 0 < forward_position < total_slots - 1:
                            position_tags.append("middle option, neither first nor last")

                        # Combine all tags into rich description for embedding
                        full_description = (
                            f"option {i+1}, position {i+1} from top, "
                            f"time {lbl}, "
                            f"{'; '.join(position_tags)}"
                        )

                        _descs.append(full_description)

                    _u_vec = embed_text(text)
                    _d_vecs = [embed_text(d) for d in _descs]
                    _scores = [float(_st_util.cos_sim(_u_vec, dv)) for dv in _d_vecs]

                    _best = max(range(len(_scores)), key=lambda i: _scores[i])
                    _sorted = sorted(_scores, reverse=True)
                    _gap = (_sorted[0] - _sorted[1]) if len(_sorted) > 1 else 1.0

                    if _scores[_best] >= 0.15 and _gap >= 0.005:
                        picked = slots[_best]
                except Exception:
                    pass

            # STEP 6: Single slot + any confirmation
            if not picked and len(slots) == 1 and is_confirm(text):
                picked = slots[0]

            if not picked:
                return "Which of these works -- " + ", ".join(slot_labels) + "?"

            self.session.time = picked
            self.session.awaiting_slot_pick = False
            self.session.awaiting_confirmation = True

            # Route based on which flow owns this slot pick
            if self.session.awaiting_reschedule_confirm or self.session.awaiting_reschedule:
                # Clear slot pick but keep reschedule confirm active
                self.session.awaiting_slot_pick = False
                self.session.awaiting_confirmation = False
                self.session.awaiting_reschedule_confirm = True
                return (
                    f"Move your appointment to {self.session.date} at {_slot_label(picked)}? "
                    "Should I go ahead?"
                )
            # Normal booking flow
            self.session.awaiting_confirmation = True
            return f"Got it -- {self.session.date} at {_slot_label(picked)}. Should I book that?"

        # USER DETAILS (new bookings only)
        if self.session.awaiting_user_details:
            name, email, phone = extract_user_details(text)

            # Collect name first if missing
            if not self.session.name:
                if not name:
                    return "What's your name?"
                self.session.name = name

                # If we ONLY got name (no email yet), ask for email
                if not self.session.email and not email:
                    return f"Got it, {self.session.name}! What email address should I use for the booking?"

            # Collect email next if missing
            if not self.session.email:
                if not email:
                    return f"Got it, {self.session.name}! What email address should I use for the booking?"
                self.session.email = email
                self.session.phone = phone or "0000000000"

            # Both present -- book
            self.session.awaiting_user_details = False
            return self._run_booking()

        # CONFIRMATION
        if self.session.awaiting_confirmation:
            # New time given -> update and re-ask
            _new_date, _new_time = extract_datetime(text)
            _has_correction = bool(_new_time or _new_date)
            _is_correcting = _has_correction and (
                _new_time != self.session.time
                or (_new_date and _new_date != self.session.date)
            )
            if _is_correcting:
                if _new_date:
                    self.session.date = _new_date
                if _new_time:
                    self.session.time = _new_time
                services_str = ", ".join(self.session.items_selected) if self.session.items_selected else "your appointment"
                return f"Updated -- {self.session.date} at {self.session.time} for {services_str}. Should I book that?"

            # Explicit confirm keywords -- is_confirm() misses "confirm booking", "go ahead", etc.
            _CONFIRM_WORDS = {"yes", "yeah", "yep", "sure", "ok", "okay", "correct", "right",
                              "confirm", "go ahead", "proceed", "do it", "book it", "book that",
                              "sounds good", "that's right", "that's correct", "yup", "absolutely"}
            is_conf = is_confirm(text) or any(w in text_l for w in _CONFIRM_WORDS)
            is_rej  = is_reject(text)
            # Don't treat a genuine service/date sentence as confirm/reject
            if len(text.split()) > 6 and not any(w in text_l for w in {"yes", "yeah", "confirm", "go ahead", "book it", "correct", "right"}):
                is_conf = False

            if is_rej:
                _resch_options = getattr(self.session, "_reschedule_options", [])
                _in_reschedule = (
                    self.session.awaiting_reschedule_confirm
                    or self.session.awaiting_reschedule
                )
                if _in_reschedule and len(_resch_options) > 1:
                    # Route back to re-pick
                    self.session.awaiting_reschedule_confirm = False
                    self.session.awaiting_confirmation = False
                    self.session.awaiting_reschedule = True
                    self.session.awaiting_reschedule_pick = True
                    return self._list_bookings_for_pick(_resch_options, "reschedule")

                # Check if user also gave a new time in same message ("no, 8pm")
                _rej_date2, _rej_time2 = extract_datetime(text)
                print(f"DEBUG REJ EXTRACT: date={_rej_date2} time={_rej_time2}")
                if _rej_time2 and _rej_time2 != self.session.time:
                    if _rej_date2:
                        self.session.date = _rej_date2
                    self.session.time = _rej_time2
                    _svcs_rej = ", ".join(self.session.items_selected) if self.session.items_selected else "your appointment"
                    return f"Got it -- {self.session.date} at {_rej_time2} for {_svcs_rej}. Should I book that?"
                if self.session.suggested_slots:
                    self.session.awaiting_confirmation = False
                    self.session.awaiting_slot_pick = True
                    from datetime import datetime as _rdt
                    def _rfmt(t):
                        try: return _rdt.strptime(t, "%H:%M").strftime("%I:%M %p").lstrip("0")
                        except Exception: return t
                    return (
                        "Here are the available times -- "
                        + ", ".join(_rfmt(s) for s in self.session.suggested_slots)
                        + ". Which works?"
                    )
                self.session.awaiting_confirmation = False
                return "Sure -- what time works better for you?"

            if is_conf:
                # Reschedule confirmation -- use email from lookup + name from profile, never re-ask
                if self.session.awaiting_reschedule_confirm or self.session.awaiting_reschedule:
                    if not self.session.name:
                        _rb_conf = getattr(self.session, "_reschedule_booking", None)
                        self.session.name = (
                            self.profile.get("name")
                            or (_rb_conf.get("name") if _rb_conf else None)
                            or (_rb_conf.get("_att_name") if _rb_conf else None)
                            or "Guest"
                        )
                    if not self.session.email:
                        self.session.email = (
                            getattr(self.session, "_reschedule_email", None)
                            or self.profile.get("email", "")
                        )

                    if not getattr(self.session, "_reschedule_email", None):
                        self.session._reschedule_email = (
                            self.session.email
                            or self.profile.get("email", "")
                        )
                    self.session.awaiting_confirmation = False
                    self.session.awaiting_reschedule_confirm = False
                    self.session.awaiting_reschedule = False
                    return self._run_reschedule()

                # ========== WEEKEND CHECK - BEFORE CONTACT DETAILS ==========
                from core.time_utils import is_weekend, next_weekday
                
                if is_weekend(self.session.date, "Asia/Kolkata"):
                    next_wd = next_weekday(self.session.date, "Asia/Kolkata")
                    requested_time = self.session.time or "10:00"

                    _original_date = self.session.date
                    
                    # Fetch available slots for the next weekday FIRST
                    from bookings import get_available_slots
                    from datetime import datetime as _dt_wd
                    
                    self._ensure_chosen_event()
                    try:
                        raw_slots = get_available_slots(self.session, next_wd, "Asia/Kolkata")
                        available_times = [
                            _dt_wd.fromisoformat(s).strftime("%H:%M")
                            for s in raw_slots if s
                        ]
                    except:
                        available_times = []
                    
                    def _fmt_time(t):
                        try:
                            return _dt_wd.strptime(t, "%H:%M").strftime("%I:%M %p").lstrip("0")
                        except:
                            return t
                    
                    # CHECK: Is the same time available?              
                    if requested_time in available_times:
                        # Same time IS available — ask if they want it
                        self.session._weekend_original_date  = self.session.date
                        self.session._weekend_next_weekday   = next_wd
                        self.session._weekend_requested_time = requested_time
                        self.session._weekend_all_slots      = available_times  # store all slots
                        self.session.awaiting_weekend_choice = True
                        self.session.awaiting_confirmation   = False
                        return (
                            f"{_original_date} is a weekend. The next available day is {next_wd}. "
                            f"Would you like the same time ({_fmt_time(requested_time)}) on {next_wd}?"
                        )
                    else:
                        # Same time NOT available — ask nearest or earliest
                        self.session._weekend_original_date  = self.session.date
                        self.session._weekend_next_weekday   = next_wd
                        self.session._weekend_requested_time = requested_time
                        self.session._weekend_all_slots      = available_times
                        self.session.awaiting_weekend_choice = True
                        self.session.awaiting_confirmation   = False
                        
                        if not available_times:
                            return (
                                f"{_original_date} is a weekend and {next_wd} "
                                f"has no available slots. Want to try a different date?"
                            )
                        return (
                            f"{_original_date} is a weekend. The next available day is {next_wd}. "
                            f"{_fmt_time(requested_time)} isn't available on {next_wd} — "
                            f"would you like the nearest slot to {_fmt_time(requested_time)}, "
                            f"or see the earliest available?"
                        )
                                    
                # ========== END WEEKEND CHECK ==========

                # New booking confirmation -- always ask fresh (never silently pre-fill from profile)
                if not self.session.name:
                    self.session.name = self.profile.get("name", "")
                if not self.session.email:
                    self.session.email = self.profile.get("email", "")

                if not self.session.name or not self.session.email:
                    self.session.awaiting_user_details = True
                    self.session.awaiting_confirmation = False
                    return "Almost there -- what's your name and email address?"

                return self._run_booking()

            # Ambiguous -- extract service if present then re-prompt
            _detected = extract_service(text, CATALOG)
            if _detected:
                _domain, _service = _detected
                if self.session.domain is None:
                    self.session.domain = _domain
                self.session.add_item(_service)
            services_str = ", ".join(self.session.items_selected) if self.session.items_selected else "your appointment"
            return (
                f"Just to confirm -- {self.session.date} at {self.session.time} "
                f"for {services_str}. Should I go ahead?"
            )

        # CANCEL EMAIL VERIFICATION -- look up booking live from Cal.com
        if self.session.awaiting_cancel_email:
            import re as _re
            _email_match = _re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
            entered = _email_match.group(0).strip() if _email_match else ""

            _prefill = getattr(self.session, "_cancel_prefill_email", None)
            if not entered and _prefill:
                entered = _prefill
                self.session._cancel_prefill_email = None

            # If profile email is known, offer it automatically on first ask
            _profile_email = self.profile.get("email", "")
            _cancel_attempts = getattr(self.session, "_cancel_email_attempts", 0)
            if not entered and _profile_email and _cancel_attempts == 0:
                # Silently use profile email if user just said "yes"/"same"/blank
                if is_confirm(text) or any(w in text_l for w in {"same", "that", "it", "yes", "yeah", "use"}):
                    entered = _profile_email
                else:
                    # Offer it
                    self.session._cancel_email_attempts = -1  # flag: we already offered
                    return f"Is it {_profile_email}? Or type a different email."

            # User confirmed or typed email -- if still empty, ask plainly
            if not entered:
                # Check if user said "yes" / "same" after we offered profile email
                if _profile_email and any(w in text_l for w in {"yes", "yeah", "same", "that", "correct", "right", "it is", "yep"}):
                    entered = _profile_email
                elif not entered:
                    return "Could you type the email address that was used when booking?"

            # Look up bookings on Cal.com for this email
            from bookings import lookup_booking_by_email
            from datetime import datetime
            import pytz

            bookings = lookup_booking_by_email(entered)

            if not bookings:
                attempts = getattr(self.session, "_cancel_email_attempts", 0) + 1
                self.session._cancel_email_attempts = attempts
                if attempts >= 2:
                    self.session.awaiting_cancel_email = False
                    self.session._cancel_email_attempts = 0
                    return ("I could not find any upcoming bookings for that email. "
                            "Please cancel directly at cal.com or contact the salon.")
                return "I could not find a booking with that email. Please double-check and try again."

            # Found booking(s) -- match by hint, list all if ambiguous
            from datetime import datetime
            tz = pytz.timezone("Asia/Kolkata")
            hint_date = getattr(self.session, "_cancel_hint_date", None)
            hint_time = getattr(self.session, "_cancel_hint_time", None)
            self.session._cancel_email_attempts = 0
            self.session._last_cancel_email = entered
            self.session._all_looked_up_bookings = bookings  # save for same-day detection

            def _fmt_b(b):
                try:
                    import re as _re_fmtb
                    _fst = _re_fmtb.sub(r"\.\d+Z$","Z",b["start_time"]).replace(".000Z","Z").replace("Z","+00:00")
                    bs = datetime.fromisoformat(_fst).astimezone(tz)
                    return _fmt_booking(b)
                except Exception:
                    return b.get("start_time","")

            # Try date/time hint match
            matched = None
            if hint_date:
                for b in bookings:
                    try:
                        import re as _re_hint
                        _st_hint = _re_hint.sub(r"\.\d+Z$", "Z", b["start_time"]).replace(".000Z","Z").replace("Z","+00:00")
                        bs = datetime.fromisoformat(_st_hint).astimezone(tz)
                        if bs.strftime("%Y-%m-%d") == hint_date:
                            if hint_time is None or bs.strftime("%H:%M") == hint_time:
                                matched = b
                                break
                    except Exception:
                        continue

            def _set_pending(b):
                """Stage a booking for confirmation without committing."""
                self.session._pending_cancel_id  = b["id"]
                self.session._pending_cancel_uid = b["uid"]

            def _check_past(b):
                """Returns (is_hard_blocked, is_grace_period, minutes_elapsed)"""
                try:
                    from core.time_utils import minutes_since_start, CANCEL_GRACE_MINUTES
                    mins = minutes_since_start(b["start_time"])
                    if mins > CANCEL_GRACE_MINUTES:
                        return True, False, mins
                    if mins > 0:
                        return False, True, mins
                except Exception:
                    pass
                return False, False, 0

            if matched:
                _hard, _grace, _mins = _check_past(matched)
                if _hard:
                    _mins_int = int(_mins)
                    return (f"That appointment started {_mins_int} minutes ago and can no longer be cancelled here. "
                            f"Please contact the salon directly. If you'd like to reschedule instead, just say reschedule.")
                _set_pending(matched)
                self.session.awaiting_cancel_email = False
                self.session.awaiting_cancel_confirm = True
                others = [b for b in bookings if b["id"] != matched["id"]]
                other_str = ""
                if others:
                    other_str = f" You also have: {'; '.join(_fmt_b(b) for b in others[:2])}."
                _grace_warning = f" [!] Note: this appointment started {int(_mins)} minutes ago." if _grace else ""
                return f"Just to confirm -- is this the booking you want to cancel?\n{_fmt_b(matched)}.{other_str}{_grace_warning}"

            elif len(bookings) == 1:
                _hard, _grace, _mins = _check_past(bookings[0])
                if _hard:
                    return (f"That appointment started {int(_mins)} minutes ago and can no longer be cancelled here. "
                            f"Please contact the salon directly. If you'd like to reschedule instead, just say reschedule.")
                _set_pending(bookings[0])
                self.session.awaiting_cancel_email = False
                self.session.awaiting_cancel_confirm = True
                _grace_warning = f" [!] Note: this appointment started {int(_mins)} minutes ago." if _grace else ""
                return f"Just to confirm -- is this the booking you want to cancel?\n{_fmt_b(bookings[0])}{_grace_warning}"

            else:
                # Multiple bookings, no clear match -- list and ask
                self.session._cancel_booking_options = bookings
                self.session.awaiting_cancel_pick = True
                self.session.awaiting_cancel_email = False
                # Clear reschedule flags so reschedule_pick block cannot hijack
                self.session.awaiting_reschedule = False
                self.session.awaiting_reschedule_pick = False
                self.session.awaiting_reschedule_confirm = False
                return "Found multiple bookings for that email. " + self._list_bookings_for_pick(bookings, "cancel")

        # CANCEL BOOKING PICK -- user choosing from multiple bookings
        if getattr(self.session, "awaiting_cancel_pick", False):
            from datetime import datetime
            import pytz as _pytz
            _tz = _pytz.timezone("Asia/Kolkata")
            options = getattr(self.session, "_cancel_booking_options", [])

            import re as _re
            picked = None

            # ── STEP 1: Explicit time match (e.g. "the one at 6", "6pm", "18:00") ──
            # Must run BEFORE semantic picker to avoid ambiguity with bare numbers
            _time_match = _re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text.lower())
            if _time_match:
                _hh = int(_time_match.group(1))
                _mm = int(_time_match.group(2) or 0)
                _ap = (_time_match.group(3) or "").lower()
                if _ap == "pm" and _hh != 12:
                    _hh += 12
                elif _ap == "am" and _hh == 12:
                    _hh = 0
                elif not _ap and _hh <= 12:
                    # No AM/PM -- try both PM and AM against available booking times
                    _cand_pm = f"{_hh + 12:02d}:{_mm:02d}" if _hh < 12 else f"{_hh:02d}:{_mm:02d}"
                    _cand_am = f"{_hh:02d}:{_mm:02d}"
                    for _opt in options:
                        try:
                            _st = _re.sub(r"\.\d+Z$", "Z", _opt["start_time"]).replace(".000Z","Z").replace("Z","+00:00")
                            _bs = datetime.fromisoformat(_st).astimezone(_tz)
                            _opt_time = _bs.strftime("%H:%M")
                            if _opt_time in (_cand_pm, _cand_am):
                                picked = _opt
                                break
                        except Exception:
                            continue
                if not picked and _ap:
                    _cand = f"{_hh:02d}:{_mm:02d}"
                    for _opt in options:
                        try:
                            _st = _re.sub(r"\.\d+Z$", "Z", _opt["start_time"]).replace(".000Z","Z").replace("Z","+00:00")
                            _bs = datetime.fromisoformat(_st).astimezone(_tz)
                            if _bs.strftime("%H:%M") == _cand:
                                picked = _opt
                                break
                        except Exception:
                            continue

            # ── STEP 2: Explicit numeric index -- "1", "2", "#2", "option 2" ──
            if not picked:
                _num_match = _re.search(
                    r"(?:^|\b(?:option|number|no\.?|#)\s*)([1-9])\b",
                    text.lower()
                )
                if _num_match:
                    _idx = int(_num_match.group(1)) - 1
                    if 0 <= _idx < len(options):
                        picked = options[_idx]

            # ── STEP 3: Semantic similarity -- handles ordinals, relative refs ──
            if not picked:
                import pytz as _cpick_ptz
                _cpick_tz = _cpick_ptz.timezone("Asia/Kolkata")
                _cpick_idx = _pick_booking_by_intent(text, options, _cpick_tz)
                if _cpick_idx is not None:
                    picked = options[_cpick_idx]

            # Could not identify -- ask again naturally
            if not picked:
                def _pick_fmt(i, b):
                    try:
                        return f"{i+1}. {_fmt_booking(b)}"
                    except Exception:
                        return f"{i+1}. {b['start_time']}"
                return "Which booking did you mean? " + self._list_bookings_for_pick(options, "cancel")

            # Check if picked booking is already past
            try:
                from core.time_utils import minutes_since_start, CANCEL_GRACE_MINUTES
                _pmins = minutes_since_start(picked["start_time"])
                if _pmins > CANCEL_GRACE_MINUTES:
                    return (f"That appointment started {int(_pmins)} minutes ago and can no longer be cancelled here. "
                            f"Please contact the salon directly or say 'reschedule' to move it.")
            except Exception:
                pass
            self.session.awaiting_cancel_pick = False
            self.session.awaiting_cancel_confirm = True
            # Clear any stale reschedule state so the new-time block cannot fire
            self.session.awaiting_reschedule = False
            self.session.awaiting_reschedule_confirm = False
            self.session.awaiting_reschedule_pick = False
            self.session._pending_cancel_id  = picked["id"]
            self.session._pending_cancel_uid = picked["uid"]
            others = [b for b in options if b["id"] != picked["id"]]
            try:
                _st_cp = _re.sub(r"\.\d+Z$", "Z", picked["start_time"]).replace(".000Z","Z").replace("Z","+00:00")
                bs = datetime.fromisoformat(_st_cp).astimezone(_tz)
                appt_str = bs.strftime("%b %d at %I:%M %p")
                svc = ", ".join(picked["services"]) if picked["services"] else "Appointment"
            except Exception:
                appt_str = picked["start_time"]
                svc = "Appointment"
            other_str = ""
            if others:
                def _fmt(b):
                    try:
                        import re as _re_fmt_inner
                        _st_fm = _re_fmt_inner.sub(r"\.\d+Z$", "Z", b["start_time"]).replace(".000Z","Z").replace("Z","+00:00")
                        s = datetime.fromisoformat(_st_fm).astimezone(_tz)
                        _svc_str = ', '.join(b['services']) if b['services'] else 'Appointment'
                        _hour = int(s.strftime("%I"))
                        _ampm = s.strftime("%p")
                        _min = s.strftime("%M")
                        return f"{_svc_str} on {s.strftime('%b')} {s.day} at {_hour}:{_min} {_ampm}"
                    except Exception:
                        return b.get("start_time", "unknown")
                other_str = f" Your other booking -- {'; '.join(_fmt(b) for b in others[:2])} -- will remain active."
            return f"Just to confirm -- cancel {svc} on {appt_str}?{other_str}"

        # CANCEL CONFIRM -- user confirms which booking to cancel
        if getattr(self.session, "awaiting_cancel_confirm", False):
            if is_reject(text):
                self.session.awaiting_cancel_confirm = False
                self.session._pending_cancel_id = None
                self.session._pending_cancel_uid = None
                _existing_options = getattr(self.session, "_cancel_booking_options", [])
                if _existing_options:
                    # Go back to pick -- don't re-ask email
                    self.session.awaiting_cancel_pick = True
                    return "No problem -- " + self._list_bookings_for_pick(_existing_options, "cancel")
                # No options stored -- fall back to email
                self.session.awaiting_cancel_email = True
                return "No problem -- could you clarify which booking you'd like to cancel? (date, time or service)"

            # Confirmed -- check for same-day other bookings
            self.session._found_booking_id  = getattr(self.session, "_pending_cancel_id", None)
            self.session._found_booking_uid = getattr(self.session, "_pending_cancel_uid", None)
            self.session.awaiting_cancel_confirm = False

            # Look for same-day bookings from the same email
            _same_day_others = []
            _pending_id = self.session._found_booking_id
            _all_bookings = getattr(self.session, "_all_looked_up_bookings", [])
            if _all_bookings and _pending_id:
                import pytz as _ptz; from datetime import datetime as _dt
                import re as _re_cd
                _tz = _ptz.timezone("Asia/Kolkata")

                # Find the date of the booking being cancelled
                _cancel_date = None
                for b in _all_bookings:
                    if b["id"] == _pending_id:
                        try:
                            _st_cd = _re_cd.sub(r"\.\d+Z$", "Z", b["start_time"]).replace(".000Z","Z").replace("Z","+00:00")
                            _cancel_date = _dt.fromisoformat(_st_cd).astimezone(_tz).strftime("%Y-%m-%d")
                        except Exception: pass
                        break

                if _cancel_date:
                    for b in _all_bookings:
                        if b["id"] == _pending_id:
                            continue
                        try:
                            _st_bd = _re_cd.sub(r"\.\d+Z$", "Z", b["start_time"]).replace(".000Z","Z").replace("Z","+00:00")
                            _bd = _dt.fromisoformat(_st_bd).astimezone(_tz).strftime("%Y-%m-%d")
                            if _bd == _cancel_date:
                                _same_day_others.append(b)
                        except Exception: pass

            if _same_day_others:
                def _fmtsd(b):
                    import pytz as _ptz; from datetime import datetime as _dt
                    _tz = _ptz.timezone("Asia/Kolkata")
                    try:
                        import re as _re_cf
                        _cst = _re_cf.sub(r"\.\d+Z$","Z",b["start_time"]).replace(".000Z","Z").replace("Z","+00:00")
                        bs = _dt.fromisoformat(_cst).astimezone(_tz)
                        svc = ", ".join(b["services"]) if b["services"] else "Appointment"
                        return f"{svc} at {bs.strftime('%I:%M %p')}"
                    except Exception: return b["start_time"]
                other_labels = " and ".join(_fmtsd(b) for b in _same_day_others[:2])
                self.session._same_day_cancel_others = _same_day_others
                self.session.awaiting_cancel_also_others = True
                self.session.awaiting_cancel_reason = False
                return f"You also have {other_labels} on the same day. Would you like to cancel that too, or just keep it?"

            self.session.awaiting_cancel_reason = True
            return ("Sure, cancelling. Would you mind sharing the reason? "
                    "It's completely optional -- just helps the salon.")

        # CANCEL SAME-DAY OTHERS -- user decides whether to cancel other same-day bookings
        if getattr(self.session, "awaiting_cancel_also_others", False):
            self.session.awaiting_cancel_also_others = False
            _others = getattr(self.session, "_same_day_cancel_others", [])
            _cancel_all = is_confirm(text) or any(w in text.lower() for w in ["both", "all", "yes cancel", "cancel both", "cancel all", "cancel them"])

            if _cancel_all and _others:
                # Queue all for cancellation -- store extra ids
                self.session._extra_cancel_ids  = [b["id"]  for b in _others]
                self.session._extra_cancel_uids = [b["uid"] for b in _others]
                self.session.awaiting_cancel_reason = True
                return "Got it -- cancelling both. Would you like to share a reason, or should I just go ahead?"
            else:
                # Keep the other booking -- only cancel the main one; ask reason
                self.session._extra_cancel_ids  = []
                self.session._extra_cancel_uids = []
                self.session.awaiting_cancel_reason = True
                return "Sure, keeping the other booking. Would you like to share a reason for cancelling this one, or should I just go ahead?"

        # CANCEL REASON COLLECTION
        if self.session.awaiting_cancel_reason:
            # Check if user is correcting the booking (date/time in response) rather than giving a reason
            _corr_date, _corr_time = extract_datetime(text)
            _correction_words = any(w in text.lower() for w in ["tomorrow", "today", "march", "april", "11th", "12th", "wrong", "not that", "different", "other", "no i said", "i said for"])
            if (_corr_date or _correction_words) and not any(w in text.lower() for w in ["because", "reason", "meeting", "appointment", "cancel", "busy", "work", "personal"]):
                # User is correcting the date -- update hint and re-pick
                if _corr_date:
                    self.session._cancel_hint_date = _corr_date
                if _corr_time:
                    self.session._cancel_hint_time = _corr_time
                # Re-run lookup with updated hints
                from bookings import lookup_booking_by_email
                _stored_email = getattr(self.session, "_last_cancel_email", None)
                if _stored_email:
                    _bookings = lookup_booking_by_email(_stored_email)
                    import pytz as _pytz
                    from datetime import datetime as _dt
                    _tz = _pytz.timezone("Asia/Kolkata")
                    _hint_d = self.session._cancel_hint_date
                    _hint_t = self.session._cancel_hint_time
                    _picked = _bookings[0] if _bookings else None
                    import re as _re_cr
                    for _b in _bookings:
                        try:
                            _st_cr = _re_cr.sub(r"\.\d+Z$", "Z", _b["start_time"]).replace(".000Z","Z").replace("Z","+00:00")
                            _bs = _dt.fromisoformat(_st_cr).astimezone(_tz)
                            if _bs.strftime("%Y-%m-%d") == _hint_d:
                                if _hint_t is None or _bs.strftime("%H:%M") == _hint_t:
                                    _picked = _b
                                    break
                        except Exception:
                            continue
                    if _picked:
                        self.session._found_booking_id = _picked["id"]
                        self.session._found_booking_uid = _picked["uid"]
                        try:
                            _st_crp = _re_cr.sub(r"\.\d+Z$", "Z", _picked["start_time"]).replace(".000Z","Z").replace("Z","+00:00")
                            _bs = _dt.fromisoformat(_st_crp).astimezone(_tz)
                            _appt = _bs.strftime("%B %d at %I:%M %p")
                        except Exception:
                            _appt = _picked["start_time"]
                        _svcs = ", ".join(_picked["services"]) if _picked["services"] else "your appointment"
                        return f"Got it -- switching to {_svcs} on {_appt}. Mind sharing why you are cancelling?"
                return "Which booking did you want to cancel? Could you confirm the date and time?"

            # Reason is optional -- "no", "skip", "go ahead", "proceed" all mean no reason
            _skip_words = {"no", "nope", "skip", "none", "go ahead", "proceed", "just cancel", "no reason", "prefer not", "don't want to", "dont want to", "rather not"}
            _text_l = text.strip().lower()
            if not text.strip() or _text_l in _skip_words or any(w in _text_l for w in ["go ahead", "just cancel", "no reason", "prefer not", "rather not", "dont want", "don't want"]):
                reason = "No reason provided"
            else:
                reason = text.strip()
            self.session.awaiting_cancel_reason = False
            self.session.awaiting_slot_pick = False
            self.session.awaiting_confirmation = False
            self.session.awaiting_reschedule = False
            self.session.awaiting_reschedule_confirm = False
            import os, requests
            api_key = os.getenv("CAL_API_KEY")
            # Prefer live-looked-up booking id over profile (handles multi-user scenarios)
            booking_uid = getattr(self.session, "_found_booking_uid", None) or self.profile.get("booking_uid")
            booking_id  = getattr(self.session, "_found_booking_id", None) or self.profile.get("booking_id")
            cancel_ref  = booking_uid or booking_id  # uid preferred; fallback to numeric id
            url = f"https://api.cal.com/v2/bookings/{cancel_ref}/cancel"
            print(f"CANCEL using uid={booking_uid} id={booking_id} ref={cancel_ref}")
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            resp = requests.post(url, json={"cancellationReason": reason}, headers=headers)
            print(f"CANCEL API STATUS: {resp.status_code}")
            print(f"CANCEL API RESPONSE: {resp.text[:300]}")
            if resp.status_code in [200, 201, 404]:
                # Also cancel same-day extras if user chose to cancel all
                _extra_uids = getattr(self.session, "_extra_cancel_uids", [])
                _extra_ids  = getattr(self.session, "_extra_cancel_ids", [])
                _cancelled_extra = 0
                for _eref in (_extra_uids if _extra_uids else _extra_ids):
                    _er = requests.post(
                        f"https://api.cal.com/v2/bookings/{_eref}/cancel",
                        json={"cancellationReason": reason}, headers=headers)
                    if _er.status_code in [200, 201]:
                        _cancelled_extra += 1
                    print(f"EXTRA CANCEL {_eref}: {_er.status_code}")
                clear_profile(email=self.session.email)
                clear_session()
                self.session.reset()
                self.recommendation_made = False
                self.session.awaiting_rebook_after_cancel = True
                _extra_msg = f" ({_cancelled_extra + 1} bookings cancelled)" if _cancelled_extra else ""
                return f"Done -- your booking has been cancelled{_extra_msg}. Would you like to book a new appointment?"
            # API failed -- save reason so we can retry without asking again
            self.session.pending_cancel_reason = reason
            self.session.awaiting_cancel_retry = True
            self._save()
            return "Something went wrong on our end. Want me to try cancelling again?"

        # CANCEL RETRY -- user said yes to retry after API failure
        if self.session.awaiting_cancel_retry:
            if is_confirm(text):
                self.session.awaiting_cancel_retry = False
                import os, requests
                api_key = os.getenv("CAL_API_KEY")
                booking_uid = getattr(self.session, "_found_booking_uid", None) or self.profile.get("booking_uid")
                booking_id  = getattr(self.session, "_found_booking_id", None) or self.profile.get("booking_id")
                cancel_ref  = booking_uid or booking_id
                print(f"RETRY using uid={booking_uid} id={booking_id} ref={cancel_ref}")
                reason = self.session.pending_cancel_reason or "No reason provided"
                self.session.pending_cancel_reason = None
                url = f"https://api.cal.com/v2/bookings/{cancel_ref}/cancel"
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                resp = requests.post(url, json={"cancellationReason": reason}, headers=headers)
                print(f"RETRY CANCEL STATUS: {resp.status_code}")
                print(f"RETRY CANCEL RESPONSE: {resp.text[:300]}")
                if resp.status_code in [200, 201, 404]:
                    clear_profile(email=self.session.email)
                    clear_session()
                    self.session.reset()
                    self.recommendation_made = False
                    self.session.awaiting_rebook_after_cancel = True
                    return "Done -- your booking has been cancelled. Would you like to book a new appointment?"
                self.session.awaiting_cancel_retry = True
                self.session.pending_cancel_reason = reason
                self._save()
                return "Still having trouble cancelling. Please try again in a moment."
            else:
                self.session.awaiting_cancel_retry = False
                self.session.pending_cancel_reason = None
                return "No problem -- your booking is still active. Anything else I can help with?"

        # REBOOK AFTER CANCEL
        if self.session.awaiting_rebook_after_cancel:
            if has_cancel_keyword(text):
                self.session.awaiting_rebook_after_cancel = False
                # Fall through to cancel handling below
            elif is_reschedule_intent(text) or any(w in text_l for w in
                                                {"reschedule", "rebook", "move",
                                                    "change my", "shift"}):
                # User wants to reschedule — clear flag and fall through
                # to reschedule trigger block below
                self.session.awaiting_rebook_after_cancel = False
                # Fall through to standard flow
            elif is_confirm(text) or any(w in text_l for w in
                                        {"yes", "yeah", "sure", "ok", "okay",
                                        "book", "appointment", "yep"}):
                self.session.awaiting_rebook_after_cancel = False
                
                _reuse_email = getattr(self.session, "_last_cancel_email", None) or self.profile.get("email", "")
                _reuse_name  = self.profile.get("name", "") or self.session.name or ""

                if _reuse_email and not self.session.email:
                    self.session.email = _reuse_email
                if _reuse_name and not self.session.name:
                    self.session.name = _reuse_name
                if not self.session.phone:
                    self.session.phone = self.profile.get("phone", "0000000000")

                # ── If we have name + email, skip identity questions ──
                if self.session.email and self.session.name:
                    return "Great -- what date, time and service would you like?"
                elif self.session.email:
                    return "Great -- what's your name for the booking?"
                else:
                    return "Great -- what date, time and service would you like?"

            else:
                self.session.awaiting_rebook_after_cancel = False
                return "No problem. Let me know if you need anything else."
        # ======================================================================
        # STANDARD FLOW
        # ======================================================================

        # CANCEL -- checked first, before date/time extraction, so bare "cancel" always reaches it
        # Skip if already mid-cancel-flow (retry, reason, email) -- those priority blocks handle it
        _cancel_flow_active = (
            self.session.awaiting_cancel_retry
            or self.session.awaiting_cancel_reason
            or self.session.awaiting_cancel_email
            or self.session.awaiting_rebook_after_cancel
            or getattr(self.session, "awaiting_cancel_confirm", False)
            or getattr(self.session, "awaiting_cancel_pick", False)
            or self.session.awaiting_reschedule  # never let "cancel" keyword hijack a reschedule flow
        )
        # Never let cancel detection hijack a reschedule message
        _is_reschedule_msg = is_reschedule_intent(text)
        if has_cancel_keyword(text) and not _cancel_flow_active and not _is_reschedule_msg \
                and not getattr(self.session, "_awaiting_cancel_current_confirm", False) \
                and not getattr(self.session, "_awaiting_cancel_same_email", False):

            _have_session_booking_c = (
                self.session.name and self.session.email and self.session.booking_id
            )

            if _have_session_booking_c:
                # Ongoing session -- confirm if user means the current booking
                _cur_svcs_c = ", ".join(self.session.items_selected) if self.session.items_selected else "your appointment"
                _cur_date_c = self.profile.get("booking_date") or self.session.date or ""
                _cur_time_c = self.profile.get("booking_time") or self.session.time or ""
                self.session._awaiting_cancel_current_confirm = True
                # Stage the current booking for cancel
                self.session._pending_cancel_id  = self.session.booking_id
                self.session._pending_cancel_uid = getattr(self.session, "booking_uid", None) or self.profile.get("booking_uid")
                self.session._last_cancel_email  = self.session.email
                # Clear reschedule state
                self.session.awaiting_reschedule = False
                self.session.awaiting_reschedule_pick = False
                self.session.awaiting_reschedule_confirm = False
                _cdate, _ctime = extract_datetime(text)
                self.session._cancel_hint_date = _cdate
                self.session._cancel_hint_time = _ctime
                return (f"Would you like to cancel your {_cur_svcs_c} "
                        f"on {_cur_date_c} at {_cur_time_c}?")
            else:
                # New session -- ask for email
                self.session.awaiting_cancel_email = True
                self.session._found_booking_id = None
                self.session._found_booking_uid = None
                self.session.awaiting_reschedule = False
                self.session.awaiting_reschedule_pick = False
                self.session.awaiting_reschedule_confirm = False
                self.session.awaiting_reschedule_email = False
                self.session._reschedule_booking = None
                self.session._reschedule_options = []
                self.session._reschedule_email = None
                _cdate, _ctime = extract_datetime(text)
                self.session._cancel_hint_date = _cdate
                self.session._cancel_hint_time = _ctime
                return "Sure -- what email address was used when booking the appointment?"

        intent, confidence = detect_intent(text)
        date, time = extract_datetime(text)
        if date and time and not _is_reschedule_msg:
            intent = "book_appointment"
            # NOTE: do NOT clear awaiting_reschedule here -- reschedule messages
            # contain the booking date/time and would incorrectly wipe mid-flow state.

        # INTENT GUARD
        from core.intent_guard import guard_intent
        blocked = guard_intent(intent, text, self.session)
        if blocked:
            return blocked

        # SLOT INQUIRY -- "which slots are available", "what times are open", etc.
        # Must be checked BEFORE reschedule/cancel so it never gets misrouted
        from core.slot_inquiry_classifier import is_slot_inquiry as _classify_slot_inquiry

        _is_slot_inquiry = False
        if (not self.session.awaiting_confirmation
                and not self.session.awaiting_slot_pick
                and not has_cancel_keyword(text)
                and not (date and time)):
            _is_slot_inquiry = _classify_slot_inquiry(text)

        if _is_slot_inquiry:
            # Extract and store service from this message if not already set
            if not self.session.items_selected:
                _sq_svc = extract_service(text, CATALOG)
                if _sq_svc:
                    self.session.domain = _sq_svc[0]
                    self.session.add_item(_sq_svc[1])

            # Fetch available slots for today (or session date if set) and list them
            _sq_date = self.session.date or self.today
            _sq_time = self.session.time or "10:00"
            self.session.date = _sq_date
            try:
                from bookings import get_available_slots, check_slots
                import datetime as _sqdt
                from utils import local_to_utc
                self._ensure_chosen_event()
                raw = get_available_slots(self.session, _sq_date, "Asia/Kolkata")
                all_times = [
                    _sqdt.datetime.fromisoformat(s).strftime("%H:%M")
                    for s in raw if s
                ]
                def _sq_t2m(t):
                    h, m = map(int, t.split(":"))
                    return h * 60 + m
                if all_times:
                    try:
                        _req_sq = _sq_t2m(_sq_time)
                        nearest = sorted(all_times, key=lambda t: abs(_sq_t2m(t) - _req_sq))[:3]
                    except Exception:
                        nearest = all_times[:3]
                    def _sqfmt(t):
                        try:
                            import datetime as _sqdt2
                            return _sqdt2.datetime.strptime(t, "%H:%M").strftime("%I:%M %p").lstrip("0")
                        except Exception:
                            return t
                    _sq_label = f"on {_sq_date}" if self.session.date else "today"
                    self.session.suggested_slots = nearest      
                    self.session.awaiting_slot_pick = True      
                    return (
                        "Here are the nearest available slots "
                        + _sq_label + " -- "
                        + ", ".join(_sqfmt(s) for s in nearest)
                        + ". Which one works for you?"
                    )
                     
                else:
                    return f"There are no open slots available {('on ' + _sq_date) if self.session.date else 'today'}. Want to try a different date?"
            except Exception as _sq_err:
                return "Let me know what time works for you and I'll check if it's available."

        # RESCHEDULE TRIGGER -- checked BEFORE cancel so "reschedule my booking" never routes to cancel
        # Note: guard removed for `not (date and time)` -- "reschedule X at 7pm" should also trigger
        if _is_reschedule_msg and not self.session.awaiting_reschedule \
                and not getattr(self.session, "awaiting_reschedule_pick", False) \
                and not getattr(self.session, "awaiting_reschedule_confirm", False) \
                and not self.session.awaiting_cancel_email \
                and not getattr(self.session, "awaiting_cancel_pick", False) \
                and not getattr(self.session, "awaiting_cancel_confirm", False) \
                and not self.session.awaiting_cancel_reason \
                and not getattr(self.session, "_awaiting_reschedule_current_confirm", False) \
                and not getattr(self.session, "_awaiting_reschedule_same_email", False):

            _have_session_booking = (
                self.session.name and self.session.email and self.session.booking_id
            )

            if _have_session_booking:
                # Ongoing session -- confirm if user means the current booking
                _cur_svcs = ", ".join(self.session.items_selected) if self.session.items_selected else "your appointment"
                _cur_date = self.profile.get("booking_date") or self.session.date or ""
                _cur_time = self.profile.get("booking_time") or self.session.time or ""
                self.session._awaiting_reschedule_current_confirm = True
                # Store current booking details for use if confirmed
                self.session._reschedule_email = self.session.email
                self.session._reschedule_booking = {
                    "id":  self.session.booking_id,
                    "uid": getattr(self.session, "booking_uid", None) or self.profile.get("booking_uid"),
                    "start_time": "",
                    "services": list(self.session.items_selected),
                }
                return (f"Would you like to reschedule your {_cur_svcs} "
                        f"on {_cur_date} at {_cur_time}?")
            else:
                # New session -- ask for email directly
                self.session.awaiting_reschedule = True
                _pe = self.profile.get("email", "")
                if _pe:
                    from bookings import lookup_booking_by_email
                    _pre_bookings = lookup_booking_by_email(_pe)
                    if len(_pre_bookings) > 1:
                        import pytz as _ptz; from datetime import datetime as _dt
                        _tz = _ptz.timezone("Asia/Kolkata")
                        self.session._reschedule_email = _pe
                        self.session._reschedule_options = _pre_bookings
                        self.session.awaiting_reschedule_email = False
                        self.session.awaiting_reschedule_pick = True
                        return self._list_bookings_for_pick(_pre_bookings, "reschedule")
                    elif len(_pre_bookings) == 1:
                        self.session._reschedule_email = _pe
                        self.session._reschedule_booking = _pre_bookings[0]
                        self.session.awaiting_reschedule_email = False
                        if time:
                            self.session.time = time
                            if date:
                                self.session.date = date
                            _svcs2 = ", ".join(_pre_bookings[0].get("services", [])) or "your appointment"
                            self.session.awaiting_reschedule = False
                            self.session.awaiting_reschedule_confirm = True
                            self.session.awaiting_confirmation = True
                            return f"Got it -- reschedule to {self.session.date} at {time} for {_svcs2}. Shall I go ahead?"
                        return f"Found it -- {_fmt_booking(_pre_bookings[0])}. What date and time would you like to move it to?"
                self.session.awaiting_reschedule_email = True
                return "Sure -- what email was the booking made with?"

        # CANCEL BOOKING (semantic path -- only reached if NOT a reschedule message)
        if is_cancel_intent(text) and not (date and time) and not _cancel_flow_active and not _is_reschedule_msg:
            self.session.awaiting_cancel_email = True
            self.session._found_booking_id = None
            self.session._found_booking_uid = None
            # Clear ALL reschedule state
            self.session.awaiting_reschedule = False
            self.session.awaiting_reschedule_pick = False
            self.session.awaiting_reschedule_confirm = False
            self.session.awaiting_reschedule_email = False
            self.session._reschedule_booking = None
            self.session._reschedule_options = []
            self.session._reschedule_email = None
            self.session._cancel_hint_date = date
            self.session._cancel_hint_time = time
            return "Sure -- what email address was used when booking the appointment?"

        # RESCHEDULE EMAIL COLLECTION
        if self.session.awaiting_reschedule \
        and getattr(self.session, "awaiting_reschedule_email", False) \
        and not self.session.awaiting_cancel_email \
        and not getattr(self.session, "awaiting_cancel_pick", False) \
        and not getattr(self.session, "awaiting_cancel_confirm", False) \
        and not self.session.awaiting_cancel_reason:
            import re as _re
            _em = _re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
            _email = _em.group(0).strip() if _em else ""

            # Offer profile email if no email in message
            _profile_email_r = self.profile.get("email", "")
            _resch_attempts = getattr(self.session, "_resch_email_attempts", 0)
            if not _email and _profile_email_r and _resch_attempts == 0:
                if is_confirm(text) or any(w in text_l for w in {"same", "yes", "yeah", "use", "that", "it"}):
                    _email = _profile_email_r
                else:
                    self.session._resch_email_attempts = -1
                    return f"Is it {_profile_email_r}? Or type a different email."

            if not _email:
                if _profile_email_r and any(w in text_l for w in {"yes", "yeah", "same", "that", "correct", "right", "yep"}):
                    _email = _profile_email_r
                elif not _email:
                    self.session._resch_email_attempts = getattr(self.session, "_resch_email_attempts", 0) + 1
                    return "Could you type the email address used for the booking?"

            from bookings import lookup_booking_by_email
            _bookings = lookup_booking_by_email(_email)
            if not _bookings:
                # Try profile email as fallback
                if _profile_email_r and _email != _profile_email_r:
                    _bookings2 = lookup_booking_by_email(_profile_email_r)
                    if _bookings2:
                        _email = _profile_email_r
                        _bookings = _bookings2
                if not _bookings:
                    return (f"I couldn't find any bookings for {_email}. "
                            "Please double-check -- or spell it out letter by letter if needed.")
            self.session._reschedule_email = _email
            self.session.awaiting_reschedule_email = False
            # If one booking -- confirm it; if multiple -- list them
            import pytz as _ptz
            from datetime import datetime as _dt
            _tz = _ptz.timezone("Asia/Kolkata")
            def _fmtb(b):
                try:
                    import re as _re_fmt
                    _st = b["start_time"]
                    _st = _re_fmt.sub(r"\.\d+Z$", "Z", _st).replace(".000Z","Z")
                    _st = _st.replace("Z", "+00:00")
                    bs = _dt.fromisoformat(_st).astimezone(_tz)
                    return _fmt_booking(b)
                except Exception as _fe:
                    return b.get("start_time", "unknown time")
            if len(_bookings) == 1:
                self.session._reschedule_booking = _bookings[0]
                self.session.awaiting_reschedule_pick = False
                return f"Found it -- {_fmtb(_bookings[0])}. What date and time would you like to move it to?"
            else:
                self.session._reschedule_options = _bookings
                self.session.awaiting_reschedule_pick = True
                return self._list_bookings_for_pick(_bookings, "reschedule")

        # RESCHEDULE PICK -- user choosing which booking to reschedule
        if self.session.awaiting_reschedule and getattr(self.session, "awaiting_reschedule_pick", False) and not getattr(self.session, "awaiting_cancel_pick", False):
            self.session.suggested_combo = None

            import re as _re
            import pytz as _ptz; from datetime import datetime as _dt
            _tz = _ptz.timezone("Asia/Kolkata")
            options = getattr(self.session, "_reschedule_options", [])

            # Extract ALL explicit times (am/pm or colon only -- avoids matching bare numbers)
            _all_times = _re.findall(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)|(\d{1,2}):(\d{2})", text.lower())
            _parsed_times = []
            for _match in _all_times:
                _h = _match[0] or _match[3]
                _m = _match[1] or _match[4]
                _ap = _match[2]
                if not _h: continue
                try:
                    _hh = int(_h); _mm = int(_m) if _m else 0
                    if _ap == "pm" and _hh != 12: _hh += 12
                    if _ap == "am" and _hh == 12: _hh = 0
                    _parsed_times.append(f"{_hh:02d}:{_mm:02d}")
                except Exception: pass

            picked = None

            # ----------------------------------------------------------------
            # STEP 1: Explicit numeric index -- "1", "2", "3", "#1", "option 2"
            # This must run BEFORE semantic similarity for reliable ordinal picks
            # ----------------------------------------------------------------
            _num_match = _re.search(
                r"(?:^|\b(?:option|number|no\.?|#)\s*)([1-9])\b",
                text_l
            )
            if not _num_match and "am" not in text_l and "pm" not in text_l and ":" not in text_l:
                # bare single digit only when no time context present
                _num_match = _re.search(r"(?<!\d)([1-9])(?!\d)", text_l)
            if _num_match:
                _idx = int(_num_match.group(1)) - 1
                if 0 <= _idx < len(options):
                    picked = options[_idx]

            # ----------------------------------------------------------------
            # STEP 2: Ordinal words -- "first", "second", "third", etc.
            # ----------------------------------------------------------------
            if not picked:
                _pick_tz_s2 = _ptz.timezone("Asia/Kolkata")
                _pick_idx_s2 = _pick_booking_by_intent(text, options, _pick_tz_s2)
                print(f"DEBUG RESCHEDULE PICK RESULT: selected index={_pick_idx_s2}")
                _is_pure_digit = bool(_re.fullmatch(r"\s*\d+\s*", text_l))
                if _pick_idx_s2 is not None and not _is_pure_digit:
                    picked = options[_pick_idx_s2]

            # Could not identify -- ask again naturally
            if not picked:
                return "I didn't quite catch which one. " + self._list_bookings_for_pick(options, "reschedule")

            self.session.awaiting_reschedule_pick = False
            self.session._reschedule_booking = picked

            # Get picked booking's own time to distinguish from any target time
            _picked_time = None
            try:
                bs = _dt.fromisoformat(
                    _re.sub(r"\.\d+Z$", "Z", picked["start_time"]).replace(".000Z","Z").replace("Z","+00:00")
                ).astimezone(_tz)
                _picked_time = bs.strftime("%H:%M")
                _picked_date_str = bs.strftime("%Y-%m-%d")
                _label = _fmt_booking(picked)
            except Exception:
                _picked_date_str = None
                _label = picked["start_time"]

            # Load services from picked booking
            if not self.session.items_selected and picked.get("services"):
                for _s in picked.get("services", []):
                    self.session.add_item(_s)

            return f"Got it -- {_label}. What date and time would you like to reschedule it to?"

        # RESCHEDULE: user gives new date/time
        if self.session.awaiting_reschedule \
        and not getattr(self.session, "awaiting_reschedule_email", False) \
        and not getattr(self.session, "awaiting_reschedule_pick", False) \
        and not getattr(self.session, "awaiting_cancel_pick", False) \
        and not getattr(self.session, "awaiting_cancel_confirm", False) \
        and not self.session.awaiting_cancel_email \
        and not self.session.awaiting_cancel_reason:
            import re as _re2
            import pytz as _ptz2
            from datetime import datetime as _dt2

            # GUARD: if user is trying to cancel, don't treat as reschedule input
            if has_cancel_keyword(text) or any(w in text_l for w in
                                            {"just cancel", "no cancel", "only cancel",
                                                "don't reschedule", "dont reschedule",
                                                "not reschedule", "cancel it", "cancel that",
                                                "cancel the"}):
                _rb_cancel = getattr(self.session, "_reschedule_booking", None)
                _rs_email   = getattr(self.session, "_reschedule_email", None)

                self.session.awaiting_reschedule = False
                self.session.awaiting_reschedule_confirm = False
                self.session.awaiting_reschedule_pick = False
                self.session._reschedule_booking = None
                self.session._reschedule_options = []

                # If we already know the booking from the reschedule lookup,
                # jump straight to cancel-confirm instead of re-asking for email.
                if _rb_cancel and _rb_cancel.get("id") and _rb_cancel.get("uid"):
                    self.session._found_booking_id  = _rb_cancel["id"]
                    self.session._found_booking_uid = _rb_cancel["uid"]
                    self.session._last_cancel_email = _rs_email or self.profile.get("email", "")
                    # Store all looked-up bookings so same-day detection works
                    if _rs_email:
                        from bookings import lookup_booking_by_email as _lbe_cg
                        self.session._all_looked_up_bookings = _lbe_cg(_rs_email)
                    self.session.awaiting_cancel_email   = False
                    self.session.awaiting_cancel_confirm = True
                    return (
                        f"Just to confirm -- is this the booking you want to cancel?\n"
                        f"{_fmt_booking(_rb_cancel)}"
                    )

                # No booking locked in yet -- need email to look up
                self.session.awaiting_cancel_email = True
                self.session._found_booking_id  = None
                self.session._found_booking_uid = None
                # Pre-fill email hint from reschedule flow so cancel email block can auto-use it
                if _rs_email:
                    self.session._cancel_prefill_email = _rs_email
                _cdate, _ctime = extract_datetime(text)
                self.session._cancel_hint_date = _cdate
                self.session._cancel_hint_time = _ctime
                return "Sure -- what email address was used when booking the appointment?"

            _rb = getattr(self.session, "_reschedule_booking", None)
            if not _rb:
                return "Something went wrong -- could you tell me the email used for the booking?"

            # Determine the booking's own date so we can skip it when scanning for target
            _tz2 = _ptz2.timezone("Asia/Kolkata")
            _orig_date = None
            _orig_time = None
            try:
                _st_obs = _re2.sub(r"\.\d+Z$", "Z", _rb["start_time"]).replace(".000Z","Z").replace("Z","+00:00")
                _obs = _dt2.fromisoformat(_st_obs).astimezone(_tz2)
                _orig_date = _obs.strftime("%Y-%m-%d")
                _orig_time = _obs.strftime("%H:%M")
            except Exception:
                pass
            # Fallback: if start_time was empty (same-session reschedule),
            # pull original date/time from profile or session directly
            if not _orig_time and self.session.time:
                _orig_time = self.session.time
            if not _orig_date and self.session.date:
                _orig_date = self.session.date

            # RE-PICK GUARD
            # If the user rejects the current pick and their message contains
            # any positional / referential language (or just "the other one"),
            # treat it as a booking re-selection rather than a new date input.
            # The embedding picker handles all natural-language variants —
            # no hardcoded wordlists needed.
            _reschedule_options = getattr(self.session, "_reschedule_options", [])

            _msg_date_check, _ = extract_datetime(text)
            # "no, March 25" has a date and nothing else → treat as date update, not re-pick
            _is_pure_date_input = bool(
                _msg_date_check
                and not any(c.isalpha() for c in text_l.replace(
                    _msg_date_check, ""
                ).replace("-", "").strip())
            )

            _is_rejection = is_reject(text) or any(
                w in text_l for w in {"wrong", "not that", "actually", "wait", "different", "other"}
            )
            _is_correction = (
                _is_rejection
                and len(_reschedule_options) > 1
                and not _is_pure_date_input
            )

            if _is_correction:
                # Numeric digit check first (cheapest)
                _pick_idx_c = None
                _nm = _re2.search(r"(?<!\d)([1-9])(?!\d)", text_l)
                if _nm:
                    _ni = int(_nm.group(1)) - 1
                    if 0 <= _ni < len(_reschedule_options):
                        _pick_idx_c = _ni

                # Semantic covers everything else: ordinals, reverse-ordinals,
                # "the other one", service names, date references, etc.
                if _pick_idx_c is None:
                    _pick_idx_c = _pick_booking_by_intent(
                        text, _reschedule_options, _ptz2.timezone("Asia/Kolkata")
                    )

                if _pick_idx_c is not None:
                    _new_pick = _reschedule_options[_pick_idx_c]
                    self.session._reschedule_booking = _new_pick
                    self.session.items_selected = []
                    for _s in _new_pick.get("services", []):
                        self.session.add_item(_s)
                    return (
                        f"Got it -- {_fmt_booking(_new_pick)}. "
                        "What date and time would you like to reschedule it to?"
                    )
                else:
                    self.session.awaiting_reschedule_pick = True
                    return "Which one did you mean? " + self._list_bookings_for_pick(_reschedule_options, "reschedule")
            # ── END RE-PICK GUARD ────────────────────────────────────────────

            _rb = getattr(self.session, "_reschedule_booking", None)
            if not _rb:
                return "Something went wrong -- could you tell me the email used for the booking?"

            # Scan ALL times in message -- pick first that differs from booking's own time.
            # This correctly handles "from 7 PM to 8 PM" by skipping the booking's time.
            import re as _re_nt
            _all_times_nt = _re_nt.findall(
                r"\b(1[0-2]|0?[1-9])(?::(\d{2}))?\s*(am|pm|AM|PM)\b", text
            )
            _all_parsed_nt = []
            for _match_nt in _all_times_nt:
                try:
                    _hh = int(_match_nt[0])
                    _mm = int(_match_nt[1]) if _match_nt[1] else 0
                    _ap = _match_nt[2].lower()
                    if _ap == "pm" and _hh != 12: _hh += 12
                    if _ap == "am" and _hh == 12: _hh = 0
                    _all_parsed_nt.append(f"{_hh:02d}:{_mm:02d}")
                except Exception:
                    pass

            # Extract date first -- needed to decide if same time is acceptable
            _msg_date, _ = extract_datetime(text)
            _date_changed = bool(_msg_date and _msg_date != _orig_date)

            # Pick first time differing from booking's own time.
            # EXCEPTION: if the date is changing, allow same time
            # (user wants "same time, different day").
            _msg_time = None
            for _t_nt in _all_parsed_nt:
                if _t_nt != _orig_time:
                    _msg_time = _t_nt
                    break
                elif _date_changed:
                    _msg_time = _t_nt  # same time is fine on a different date
                    break
            # If no time found via AM/PM scan, fall back to full extract
            if not _msg_time:
                _, _msg_time_fb = extract_datetime(text)
                if _msg_time_fb and (_msg_time_fb != _orig_time or _date_changed):
                    _msg_time = _msg_time_fb

            # "same time" / "same as before" -- reuse session.time
            _same_time_phrases = {
                "same time", "same as before", "same slot", "same hour",
                "keep the time", "at the same time", "same timing",
                "same 3", "same 4", "same 5", "same 6",
                "same 7", "same 8", "same 9", "same 10", "same 11", "same 12"
            }
            _wants_same_time = any(p in text_l for p in _same_time_phrases)

            # Broad catch: "same" + "time" anywhere in message
            # handles "tomorrow at the same time", "same time next day", etc.
            if not _wants_same_time:
                _wants_same_time = ("same" in text_l and "time" in text_l)

            if _wants_same_time and not _msg_time:
                # _orig_time is the time of the booking being rescheduled -- always prefer it
                _msg_time = _orig_time or self.session.time

            # "same date" / "same day" -- reuse the booking's original date
            _same_date_phrases = {
                    "same date", "same day", "that day", "that date",
                    "the same date", "the same day", "keep the date"
            }
            _wants_same_date = any(p in text_l for p in _same_date_phrases)
            print(f"DEBUG SAME DATE: wants={_wants_same_date} _orig_date={_orig_date} _msg_date={_msg_date} text_l={text_l!r}")

            if _wants_same_date and not _msg_date:
                    _msg_date = _orig_date or self.session.date

            # Fall back to stored date if user only gives a time ("8pm")
            _eff_date = _msg_date or self.session.date or _orig_date

            # Determine effective time
            if _msg_time:
                _eff_time = _msg_time
            elif _msg_date:
                # New date given but no time -- must ask
                _eff_time = None
            else:
                # No new date, no new time -- keep whatever is in session or orig booking
                _eff_time = self.session.time or _orig_time
            
            print(f"DEBUG TIME SCAN: raw={_all_times_nt} parsed={_all_parsed_nt} orig={_orig_time} msg_time={_msg_time}")

            if _msg_time:
                self.session.time = _msg_time

            if not _eff_date:
                return "What date would you like to move it to?"

            if not self.session.date and _orig_date:
                self.session.date = _orig_date

            # Persist whatever we learned
            if _msg_date:
                self.session.date = _msg_date

            # Load services from booking if session is still empty
            if not self.session.items_selected:
                for s in _rb.get("services", []):
                    self.session.add_item(s)

            if not _eff_time and self.session.time and self.session.time != _orig_time:
                _eff_time = self.session.time

            if _eff_time:
                self.session.time = _eff_time
                _det = extract_service(text, CATALOG)
                if _det:
                    self.session.add_item(_det[1])
                svcs = ", ".join(self.session.items_selected) if self.session.items_selected else "your appointment"
                self.session.awaiting_reschedule = False
                self.session.awaiting_reschedule_confirm = True
                self.session.awaiting_confirmation = True
                return f"Got it -- reschedule to {_eff_date} at {_eff_time} for {svcs}. Shall I go ahead?"
            else:
                return f"What time on {_eff_date} works for you?"

        # -- CHANGE TIME -------------------------------------------------------
        if "change time" in text_l or "different time" in text_l:
            self.session.time = None
            self.session.awaiting_confirmation = False
            return "Of course -- what time did you have in mind?"

        # -- COMBO INQUIRY -----------------------------------------------------
        if intent == "combo_inquiry":
            if not self.session.items_selected:
                return "What service are you starting with? I can help with Haircut, Hair Spa, Massage, Facial, Manicure, or Pedicure."
            add_ons = []
            for svc in self.session.items_selected:
                for s in COMBO_SUGGESTIONS.get(svc, []):
                    if s not in self.session.items_selected and s not in add_ons:
                        add_ons.append(s)
            if not add_ons:
                return "You have a great selection already! Want to pick a time?"
            self.recommendation_made = True
            self.session.suggested_combo = add_ons[:3]
            readable = " or ".join(add_ons[:2])
            return f"We also offer {readable} that pairs well. Want to add one?"

        # -- COMBO RESPONSE ----------------------------------------------------
        if self.session.suggested_combo:
            combos = self.session.suggested_combo

            def _after_add(added_label):
                """Return next prompt after adding a service."""
                self.session.suggested_combo = None
                self.recommendation_made = True
                svcs = ", ".join(self.session.items_selected)
                if self.session.date and self.session.time:
                    self.session.awaiting_confirmation = True
                    return f"Added {added_label}! So that's {svcs} on {self.session.date} at {self.session.time}. Shall I book that?"
                elif self.session.date:
                    return f"Added {added_label}! You have {svcs}. What time works for you?"
                return f"Added {added_label}! You have {svcs}. What date and time works for you?"

            def _after_reject():
                """Return next prompt after rejecting combo."""
                self.session.suggested_combo = None
                self.recommendation_made = True
                svcs = ", ".join(self.session.items_selected)
                if self.session.date and self.session.time:
                    self.session.awaiting_confirmation = True
                    return f"No worries -- just {svcs} on {self.session.date} at {self.session.time}. Shall I book that?"
                elif self.session.date:
                    return f"No worries -- just {svcs}. What time works for you?"
                return f"No worries -- just {svcs}. What date and time works for you?"

            _reject_words = {"no", "not", "don't", "dont", "skip", "just", "stick", "only", "nope", "nah"}

            # 1. Collect ALL combo services mentioned in text (LLM rephrases "both" as "X and Y")
            _all_detected = []
            for svc in combos:
                if svc.lower() in text_l:
                    _all_detected.append(svc)
            # Also scan via extractor for aliases not caught by substring ("spa" -> "Hair Spa")
            _ext = extract_service(text, CATALOG)
            if _ext and _ext[1] in combos and _ext[1] not in _all_detected:
                _all_detected.append(_ext[1])

            # 2. "Both" / "all" explicit words -- add every suggested combo regardless
            if any(w in text_l for w in {"both", "all", "every", "each"}) and not any(w in text_l for w in _reject_words):
                for _svc in combos:
                    self.session.add_item(_svc)
                return _after_add(" and ".join(combos))

            # 3. Multiple services detected in text -- add all of them
            if len(_all_detected) >= 2:
                for _svc in _all_detected:
                    self.session.add_item(_svc)
                return _after_add(" and ".join(_all_detected))

            # 4. Single named service
            if len(_all_detected) == 1:
                self.session.add_item(_all_detected[0])
                return _after_add(_all_detected[0])

            # 5. Ordinal pick -- "the first one", "second"
            ordinal_map = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}
            for key, idx in ordinal_map.items():
                if key in text_l and idx < len(combos):
                    self.session.add_item(combos[idx])
                    return _after_add(combos[idx])

            # 6. Blanket "yes" -- add the top suggestion only
            if is_confirm(text) and not any(w in text_l for w in _reject_words):
                top = combos[0]
                self.session.add_item(top)
                return _after_add(top)

            # 7. Reject
            if is_reject(text) or any(w in text_l for w in _reject_words):
                return _after_reject()

        # -- DATE / TIME (early store) -----------------------------------------
        # Store date/time NOW before any early-return paths (combo, service)
        # so "massage tomorrow at 6pm" retains tomorrow even if combo fires first.
        if date and not self.session.awaiting_reschedule:
            self.session.date = date
        if time and not self.session.awaiting_reschedule:
            self.session.time = time

        # -- SERVICE EXTRACTION ------------------------------------------------
        # Try multi-service extraction first (handles "Haircut,Hair Spa,Massage")
        _all_svcs = extract_all_services(text, CATALOG)
        if len(_all_svcs) > 1:
            # Multiple services detected -- add all, skip combo suggestion
            for _svc_m in _all_svcs:
                self.session.add_item(_svc_m)
            self.recommendation_made = True  # suppress combo after explicit multi-list
            _added_label = ", ".join(_all_svcs)
            if self.session.date and self.session.time:
                self.session.awaiting_confirmation = True
                return f"Got it -- {_added_label} added. So that's {_added_label} on {self.session.date} at {self.session.time}. Shall I book that?"
            return f"Got it -- {_added_label} added. What date and time works for you?"
        detected = extract_service(text, CATALOG)
        if detected:
            domain, service = detected
            if self.session.domain is None:
                self.session.domain = domain
            self.session.add_item(service)

            # Suggest combo only once, and only before date is known
            if not self.recommendation_made and not (date or self.session.date):
                suggestions = COMBO_SUGGESTIONS.get(service, [])
                add_ons = [s for s in suggestions if s not in self.session.items_selected]
                if add_ons:
                    self.recommendation_made = True
                    self.session.suggested_combo = add_ons[:3]
                    readable = " or ".join(add_ons[:2])
                    return f"Got it -- {service} added. Want to also add {readable}?"

        # -- DATE / TIME (already stored above, kept for safety) ---------------
        if date:
            self.session.date = date
        if time:
            self.session.time = time

        # -- BOOKING INTENT ----------------------------------------------------
        _has_service = bool(self.session.items_selected)
        _has_date    = bool(self.session.date)
        _has_time    = bool(self.session.time)
        _booking_trigger = intent == "book_appointment" or (_has_date and _has_time)

        if _booking_trigger and not self.session.awaiting_confirmation and not self.session.awaiting_slot_pick:
            # Reschedule path
            if self.session.awaiting_reschedule_confirm:
                self.session.awaiting_reschedule_confirm = False
                return self._run_reschedule()

            # Collect service first
            if not _has_service:
                return "What service would you like? Haircut, Hair Spa, Massage, Facial, Manicure, or Pedicure."

            # Collect date
            if not _has_date:
                svcs = ", ".join(self.session.items_selected)
                return f"Got it -- {svcs}. What date works for you?"

            # Collect time
            if not _has_time:
                svcs = ", ".join(self.session.items_selected)
                return f"And what time on {self.session.date}?"

            # All collected -- weekend check BEFORE asking for confirmation
            from core.time_utils import is_weekend, next_weekday
            from bookings import get_available_slots
            from datetime import datetime as _dt_bk

            if self.session.date and is_weekend(self.session.date, "Asia/Kolkata"):
                next_wd = next_weekday(self.session.date, "Asia/Kolkata")
                requested_time = self.session.time or "10:00"

                self._ensure_chosen_event()
                try:
                    raw_slots = get_available_slots(self.session, next_wd, "Asia/Kolkata")
                    available_times = [
                        _dt_bk.fromisoformat(s).strftime("%H:%M")
                        for s in raw_slots if s
                    ]
                except Exception:
                    available_times = []

                def _fmt_t(t):
                    try:
                        return _dt_bk.strptime(t, "%H:%M").strftime("%I:%M %p").lstrip("0")
                    except Exception:
                        return t

                if requested_time in available_times:
                    # Same time IS available — ask if they want it
                    self.session._weekend_original_date  = self.session.date
                    self.session._weekend_next_weekday   = next_wd
                    self.session._weekend_requested_time = requested_time
                    self.session._weekend_all_slots      = available_times
                    self.session.awaiting_weekend_choice = True
                    self.session.awaiting_confirmation   = False
                    return (
                        f"{self.session.date} is a weekend. The next available day is {next_wd}. "
                        f"Would you like the same time ({_fmt_t(requested_time)}) on {next_wd}?"
                    )
                else:
                    _original_date_bk = self.session.date
                    self.session._weekend_original_date  = _original_date_bk
                    self.session._weekend_next_weekday   = next_wd
                    self.session._weekend_requested_time = requested_time
                    self.session._weekend_all_slots      = available_times
                    self.session.awaiting_weekend_choice = True
                    self.session.awaiting_confirmation   = False

                    if not available_times:
                        return (
                            f"{_original_date_bk} is a weekend and {next_wd} "
                            f"has no available slots. Want to try a different date?"
                        )
                    return (
                        f"{_original_date_bk} is a weekend. The next available day is {next_wd}. "
                        f"{_fmt_t(requested_time)} isn't available on {next_wd} — "
                        f"would you like the nearest slot to {_fmt_t(requested_time)}, "
                        f"or see the earliest available?"
                    )
                                

            # Not a weekend — proceed to normal confirmation
            svcs = ", ".join(self.session.items_selected)
            self.session.awaiting_confirmation = True
            return f"Okay -- {self.session.date} at {self.session.time} for {svcs}. Want me to book that?"

        # Progressive prompting -- guide user step by step
        if intent == "book_appointment":
            if not _has_service:
                return "What service would you like? Haircut, Hair Spa, Massage, Facial, Manicure, or Pedicure."
            if not _has_date:
                svcs = ", ".join(self.session.items_selected)
                return f"Got it -- {svcs}. What date works for you?"
            if not _has_time:
                svcs = ", ".join(self.session.items_selected)
                return f"And what time on {self.session.date}?"

        # -- THANKS ------------------------------------------------------------
        if intent == "thanks":
            return "Anytime!"

        # -- DEFAULT -----------------------------------------------------------
        # If we have partial state, nudge forward
        if _has_service and _has_date and not _has_time:
            return f"What time on {self.session.date} works for you?"
        if _has_service and not _has_date:
            return f"What date and time works for you?"
        return "What would you like to book? I can help with Haircut, Hair Spa, Massage, Facial, Manicure, or Pedicure."
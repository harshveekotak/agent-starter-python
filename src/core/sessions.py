class Session:
    def __init__(self, domain=None):
        self.domain = domain
        self.items_selected = []
        self.completed = False

        self.date = None
        self.time = None

        self.awaiting_confirmation = False
        self.awaiting_user_details = False
        self.ready_to_book = False

        self.name = None
        self.email = None
        self.phone = None

        self.suggested_slots = []
        self.awaiting_slot_pick = False
        self.booking_id = None

        self.awaiting_reschedule = False
        self.is_rescheduling = False

        self.pending_events = []
        self.awaiting_event_pick = False
        self.chosen_event = None

        self.recommendations = {}

        self.user_profile = {
            "preferred_services": {},
            "preferred_time_ranges": {},
            "booking_count": 0
        }

        self.suggested_combo = None

        # Post-completion same-identity prompt
        self._awaiting_same_identity = False

        # Contact reuse confirm (within same conversation)
        self._awaiting_contact_confirm = False  # ← ADD: was missing from __init__

        # Reschedule already-cancelled flag (prevents double cancel)
        self._reschedule_old_cancelled = False

        # Email attempts/context
        self._cancel_email_attempts = 0
        self._cancel_hint_date = None
        self._cancel_hint_time = None
        self._last_cancel_email = None

        # Cancel staging
        self._pending_cancel_id = None
        self._pending_cancel_uid = None
        self._found_booking_id = None
        self._found_booking_uid = None
        self._cancel_booking_options = []
        self._all_looked_up_bookings = []
        self._extra_cancel_ids = []
        self._extra_cancel_uids = []
        self.pending_cancel_reason = None

        # Cancel flags
        self.awaiting_cancel_email = False
        self.awaiting_cancel_confirm = False
        self.awaiting_cancel_reason = False
        self.awaiting_cancel_pick = False
        self.awaiting_cancel_retry = False
        self.awaiting_cancel_also_others = False
        self.awaiting_rebook_after_cancel = False

        # Reschedule fields
        self.awaiting_reschedule_email = False
        self.awaiting_reschedule_pick = False
        self.awaiting_reschedule_confirm = False
        self._reschedule_email = None
        self._reschedule_booking = None
        self._reschedule_options = []

        # Ongoing-session confirm flows
        self._awaiting_reschedule_current_confirm = False  # "reschedule THIS booking?"
        self._awaiting_reschedule_same_email = False       # "same email as before?"
        self._awaiting_cancel_current_confirm = False      # "cancel THIS booking?"
        self._awaiting_cancel_same_email = False           # "same email as before?"

        # Misc
        self.booking_uid = None

    def add_item(self, item):
        if item not in self.items_selected:
            self.items_selected.append(item)

    def reset(self):
        self.domain = None
        self.items_selected = []
        self.completed = False

        self.date = None
        self.time = None

        self.awaiting_confirmation = False
        self.awaiting_user_details = False
        self.ready_to_book = False

        self.suggested_slots = []
        self.awaiting_slot_pick = False
        self.booking_id = None

        self.name = None
        self.email = None
        self.phone = None

        self.awaiting_reschedule = False
        self.is_rescheduling = False

        self.pending_events = []
        self.awaiting_event_pick = False
        self.chosen_event = None

        self.suggested_combo = None

        # Identity / contact flags
        self._awaiting_same_identity = False
        self._awaiting_contact_confirm = False  # already in your reset — confirmed correct

        # Reschedule flags
        self._reschedule_old_cancelled = False
        self.awaiting_reschedule_email = False
        self.awaiting_reschedule_pick = False
        self.awaiting_reschedule_confirm = False
        self._reschedule_email = None
        self._reschedule_booking = None
        self._reschedule_options = []

        # Cancel flags
        self.awaiting_cancel_email = False
        self.awaiting_cancel_confirm = False
        self.awaiting_cancel_reason = False
        self.awaiting_cancel_pick = False
        self.awaiting_cancel_retry = False
        self.awaiting_cancel_also_others = False
        self.awaiting_rebook_after_cancel = False

        # Ongoing-session confirm flows
        self._awaiting_reschedule_current_confirm = False
        self._awaiting_reschedule_same_email = False
        self._awaiting_cancel_current_confirm = False
        self._awaiting_cancel_same_email = False

        # Cancel staging — clear so stale IDs don't leak into next cancel flow
        self._cancel_email_attempts = 0
        self._cancel_hint_date = None
        self._cancel_hint_time = None
        self._last_cancel_email = None
        self._pending_cancel_id = None
        self._pending_cancel_uid = None
        self._found_booking_id = None
        self._found_booking_uid = None
        self._cancel_booking_options = []
        self._all_looked_up_bookings = []
        self._extra_cancel_ids = []
        self._extra_cancel_uids = []
        self.pending_cancel_reason = None

        # Misc
        self.booking_uid = None
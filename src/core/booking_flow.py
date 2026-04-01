import os
import requests

from bookings import create_cal_booking, get_available_slots
from utils import local_to_utc
from core.cal_metadata import fetch_events
from core.time_utils import is_past_datetime, is_weekend, next_weekday


def run_booking_flow(
    session,
    name: str,
    email: str,
    phone: str,
    date: str,
    time: str,
    timezone: str = "Asia/Kolkata",
    is_reschedule: bool = False,
    old_booking_id=None,
):
    """
    Deterministic booking flow.
    Returns:
    - {"type": "BOOKED", "booking_uid": "..."}
    - {"type": "SLOT_UNAVAILABLE", "suggested_slots": [...]}
    - {"type": "SAY", "text": "..."}
    """
    print("Inside RUN_BOOKING_FLOW CALLED")

    # ----------------------------------------------------------
    # 1. Sanity checks -- must run before slot check and cancel
    # ----------------------------------------------------------
    if is_past_datetime(date, time, timezone):
        print(f"DEBUG PAST CHECK: date={date} time={time} now_utc={datetime.utcnow()}")
        return {
            "type": "SAY",
            "text": "That time has already passed. Want to try another one?"
        }

    # ----------------------------------------------------------
    # 2. Resolve Salon Appointment event FIRST
    #    Must happen before slot check -- get_available_slots
    #    needs session.chosen_event to know which event type
    #    to query slots for.
    # ----------------------------------------------------------
    events = fetch_events()
    print("ALL EVENT TITLES:", [e.get("title") for e in events])

    matching_events = [
        e for e in events
        if e.get("title", "").lower() == "salon appointment"
    ]

    if not matching_events:
        return {"type": "SAY", "text": "Salon Appointment event not found in Cal."}

    chosen_event = matching_events[0]
    session.chosen_event = chosen_event

    event_id = chosen_event.get("id")
    if not event_id:
        return {"type": "SAY", "text": "Service configuration error (missing event ID)."}

    duration = chosen_event.get("length")
    if not duration:
        return {"type": "SAY", "text": "Service configuration error (missing duration)."}

    # ----------------------------------------------------------
    # 3. Validate slot availability BEFORE cancelling anything
    #    Use Cal.com's actual slot list -- ensures session.chosen_event
    #    is set so get_available_slots can query the right event type.
    #    Never cancels old booking unless new slot is confirmed open.
    # ----------------------------------------------------------
    slot_available = False
    available_times = []
    try:
        import datetime as _bfdt
        raw_slots = get_available_slots(session, date, timezone)
        all_times = [
            _bfdt.datetime.fromisoformat(s).strftime("%H:%M")
            for s in raw_slots if s
        ]
        slot_available = time in all_times

        def _t2m(t):
            h, m = map(int, t.split(":"))
            return h * 60 + m

        try:
            _req = _t2m(time)
            available_times = sorted(all_times, key=lambda t: abs(_t2m(t) - _req))[:3]
        except Exception:
            available_times = all_times[:3]
    except Exception:
        slot_available = False  # fail safe: don't cancel if we can't verify

    if not slot_available:
        print(f"SLOT CHECK FAILED: {date} at {time} is not available")
        if available_times:
            session.suggested_slots = available_times
            session.awaiting_slot_pick = True
            session.awaiting_confirmation = False
            return {
                "type": "SLOT_UNAVAILABLE",
                "suggested_slots": available_times,
            }
        # No slots available at all on this date
        session.awaiting_slot_pick = False
        session.suggested_slots = []
        session.awaiting_confirmation = False
        _blocked_date = date   # save before clearing
        if not is_reschedule:
            session.date = None
            session.time = None
        return {
            "type": "SAY",
            "text": (
                f"There are no available slots on {_blocked_date} at all. "
                "Could you try a different date?"
            )
        }

    # ----------------------------------------------------------
    # 4. Slot is confirmed open -- now safe to cancel old booking
    # ----------------------------------------------------------
    if is_reschedule and old_booking_id:
        url = f"https://api.cal.com/v2/bookings/{old_booking_id}/cancel"
        headers = {
            "Authorization": f"Bearer {os.getenv('CAL_API_KEY')}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            url,
            json={"cancellationReason": "Rescheduled by customer"},
            headers=headers,
        )
        print(f"CANCEL OLD BOOKING STATUS: {resp.status_code} ref={old_booking_id}")
        if resp.status_code == 404:
            # Booking already gone — safe to proceed with new booking
            print(f"Old booking {old_booking_id} not found (404) — treating as already cancelled")
        elif resp.status_code not in [200, 201]:
            # Any other non-success status is a real error — do not proceed
            print(f"Cancel failed with status {resp.status_code}: {resp.text}")
            return {
                "type": "SAY",
                "text": "I could not cancel the old booking. Want to try again?"
            }

    # ----------------------------------------------------------
    # 5. Convert to UTC
    # ----------------------------------------------------------
    start_utc = local_to_utc(date, time, timezone)
    if not start_utc:
        return {
            "type": "SAY",
            "text": "I could not parse that date and time. Want to try again?"
        }

    # ----------------------------------------------------------
    # 6. Create booking
    # ----------------------------------------------------------
    print("========== DEBUG BOOKING INPUT ==========")
    print("Event ID:", event_id)
    print("Duration:", duration)
    print("Start UTC:", start_utc)
    print("Name:", name)
    print("Email:", email)
    print("Phone:", phone)
    print("Services:", session.items_selected)
    print("=========================================")

    try:
        result = create_cal_booking(
            session=session,
            event_id=event_id,
            start_utc=start_utc,
            name=name,
            email=email,
            phone=phone,
            timezone=timezone,
            services=session.items_selected,
        )
    except Exception as e:
        print("BOOKING EXCEPTION:", str(e))
        return {"type": "SAY", "text": "Something went wrong while creating the booking."}

    # ----------------------------------------------------------
    # 7. Handle result
    # ----------------------------------------------------------
    booking_id  = result.get("id")  if result else None
    booking_uid = result.get("uid") if result else None

    if not booking_id:
        print("BOOKING FAILED - result:", result)
        return {
            "type": "SAY",
            "text": (
                "Something went wrong while confirming your booking. "
                "The slot was available but the booking could not be created. "
                "Want to try again?"
            )
        }

    print("BOOKING SUCCESS - id:", booking_id)
    session.booking_id = booking_id
    session.completed  = True
    session.ready_to_book = False
    session.awaiting_slot_pick = False
    session.suggested_slots = []

    return {"type": "BOOKED", "booking_uid": booking_uid}
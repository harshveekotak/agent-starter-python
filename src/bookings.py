from datetime import datetime, timedelta
import os
import requests
import pytz
from utils import record_recommendation, local_to_utc

# ----------------------------
# SLOT CHECK
# ----------------------------

def check_slots(
    session,
    date: str,
    time: str,
    timezone: str = "Asia/Kolkata",
    duration: int | None = None
):
    """
    Checks if a slot can fit the required duration (in minutes).
    """

    # 1. Fetch real slots from Cal.com
    available_times = get_available_slots(session, date, timezone)

    # Convert Cal slot format  HH:MM
    readable_times = [
        datetime.fromisoformat(slot).strftime("%H:%M")
        for slot in available_times
        if slot
    ]

    if not readable_times:
        print("No available slots returned from Cal")
        return False, []

    # Requested time not available at all
    if time not in readable_times:
        def _t2m(t):
            h, m = map(int, t.split(':'))
            return h * 60 + m
        try:
            _req_min = _t2m(time)
            suggestions = sorted(sorted(readable_times, key=lambda t: abs(_t2m(t) - _req_min))[:3])
        except Exception:
            suggestions = readable_times[:3]
        print("Requested time:", time)
        print("Available suggestions:", suggestions)
        return False, suggestions

    # If duration is not provided, behave like old logic
    if not duration:
        return True, [time]

    # Convert chosen slot to UTC
    start_utc_str = local_to_utc(date, time, timezone)
    if not start_utc_str:
        return False, readable_times[:5]

    start_utc = datetime.fromisoformat(start_utc_str.replace("Z", "+00:00"))
    end_utc = start_utc + timedelta(minutes=duration)

    # Check if duration fits before next slot boundary
    for next_time in readable_times:
        next_utc_str = local_to_utc(date, next_time, timezone)
        if next_utc_str:
            next_utc = datetime.fromisoformat(next_utc_str.replace("Z", "+00:00"))
            if next_utc > start_utc:
                if end_utc <= next_utc:
                    return True, [time]
                break

    # Duration doesn't fit -- suggest 3 nearest slots
    def _t2m2(t):
        h, m = map(int, t.split(':'))
        return h * 60 + m
    try:
        _req_min2 = _t2m2(time)
        suggested = sorted(sorted(readable_times, key=lambda t: abs(_t2m2(t) - _req_min2))[:3])
    except Exception:
        suggested = readable_times[:3]

    record_recommendation(
        session,
        key=suggested[0] if suggested else None,
        rec_type="slot",
        signals={"requested_time": time, "date": date, "timezone": timezone, "strategy": "nearest_available"}
    )

    return False, suggested



# ----------------------------
# CAL.COM BOOKING
# ----------------------------

def create_cal_booking(
    session,
    event_id: int = None,
    start_utc: str = None,
    name: str = None,
    email: str = None,
    phone: str = None,
    timezone: str = "Asia/Kolkata",
    services: list[str] = None
):
    api_key = os.getenv("CAL_API_KEY")

    if not api_key:
        print("CAL_API_KEY missing")
        return None

    url = "https://api.cal.com/v2/bookings"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    body = {
        "eventTypeId": event_id,
        "start": start_utc,
        "timeZone": timezone,
        "language": "en",
        "attendees": [
            {
                "name": name,
                "email": email,
                "timeZone": timezone,
                "language": "en"
            }
        ],
        "responses": {
            "name": name,
            "email": email,
            "services": services or []
        },
        "metadata": {
            "phone": phone or "0000000000"
        }
    }

    print("========== CAL BOOKING REQUEST ==========")
    print(body)

    response = requests.post(url, json=body, headers=headers)

    print("BOOKING STATUS:", response.status_code)
    print("BOOKING RESPONSE:", response.text)

    if response.status_code not in [200, 201]:
        return None

    data = response.json()

    # Handle v2 envelope {"status":"success","data":{...}}
    if data.get("status") == "success" and "id" in data.get("data", {}):
        return data["data"]
    if "id" in data:
        return data
    return None

# ----------------------------
# AVAILABLE SLOTS FETCHER
# ----------------------------  

def get_available_slots(session, date: str, timezone="Asia/Kolkata"):

    if not session.chosen_event:
        print("No chosen_event set in session")
        return []
        
    url = "https://api.cal.com/v2/slots/available"

    headers = {
        "Authorization": f"Bearer {os.getenv('CAL_API_KEY')}"
    }

    tz = pytz.timezone(timezone)

    start_dt = f"{date}T00:00:00Z"
    end_dt = f"{date}T23:59:59Z"

    params = {
        "eventTypeId": session.chosen_event.get("id"),
        "startTime": start_dt,
        "endTime": end_dt,
        "timeZone": timezone
    }

    response = requests.get(url, params=params, headers=headers)
    print("CAL SLOT STATUS:", response.status_code)
    print("CAL SLOT RESPONSE:", response.text)
    print("CAL SLOT PARAMS:", params)

    if response.status_code != 200:
        print("Slot fetch failed:", response.text)
        return []

    data = response.json()

    slots = data.get("data", {}).get("slots", {})

    if isinstance(slots, dict):
        slots = slots.get(date, [])

    return [
        s.get("time")
        for s in slots
        if s.get("time")
    ]

# ----------------------------
# BOOKING LOOKUP BY EMAIL
# ----------------------------

def lookup_booking_by_email(email: str):
    """Fetch upcoming bookings for a given attendee email via Cal.com v2 API.
    Tries multiple filter strategies since attendeeEmail param support varies.
    """
    import os, requests as _req2
    from datetime import datetime as _ldt, timezone as _ltz
    api_key = os.getenv("CAL_API_KEY")
    if not api_key:
        return []

    headers = {
        "Authorization": f"Bearer {api_key}",
        "cal-api-version": "2024-08-13",
    }
    email_l = email.strip().lower()
    now_iso = _ldt.now(_ltz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _parse(b):
        import re as _re_bk
        # Attendee email -- list endpoint uses "attendees"
        att_email = ""
        att_name = ""
        for att in b.get("attendees", []):
            if att.get("email"):
                att_email = att["email"].strip().lower()
                att_name  = att.get("name", "")
                break

        # Cal.com v2 LIST endpoint uses "bookingFieldsResponses" (not "responses")
        bfr = b.get("bookingFieldsResponses") or b.get("responses") or {}
        if isinstance(bfr, list):
            bfr = {r.get("label","").lower(): r.get("value","") for r in bfr}

        resp_email = (bfr.get("email") or "").strip().lower()

        # Services: look in bookingFieldsResponses first, then any list value
        svcs = bfr.get("services") or bfr.get("Services") or []
        if not svcs:
            for v in bfr.values():
                if isinstance(v, list) and v and isinstance(v[0], str):
                    svcs = v
                    break

        # Cal.com v2 LIST endpoint uses "start" (not "startTime")
        raw_start = b.get("start") or b.get("startTime") or ""
        norm_start = _re_bk.sub(r"\.\d+Z$", "Z", raw_start).replace(".000Z","Z")

        return {
            "id":          b.get("id"),
            "uid":         b.get("uid"),
            "start_time":  norm_start,
            "end_time":    (b.get("end") or b.get("endTime") or ""),
            "status":      b.get("status", ""),
            "services":    svcs,
            "name":        bfr.get("name", "") or att_name,
            "email":       resp_email or att_email or email,
            "_att_email":  att_email,
            "_resp_email": resp_email,
            "_att_name":   att_name,
        }

    def _matches(b_parsed):
        return (b_parsed["_att_email"] == email_l or
                b_parsed["_resp_email"] == email_l)

    # Strategy 1: filter by attendeeEmail param
    try:
        resp = _req2.get(
            "https://api.cal.com/v2/bookings",
            params={"attendeeEmail": email, "take": 50},
            headers=headers,
        )
        print("LOOKUP STATUS:", resp.status_code)
        if resp.status_code == 200:
            raw = resp.json().get("data", [])
            print("LOOKUP S1 RAW COUNT:", len(raw) if isinstance(raw, list) else raw)
            if isinstance(raw, list) and raw:
                parsed = [_parse(b) for b in raw
                          if b.get("status","").upper() not in ("CANCELLED", "REJECTED", "CANCELED")
                          and not b.get("rescheduled")]
                # Keep upcoming bookings; include bookings from last 30 mins as grace period
                import re as _re_grace
                from datetime import datetime as _dt_grace, timezone as _tz_grace, timedelta as _td_grace
                _grace_iso = (_dt_grace.now(_tz_grace.utc) - _td_grace(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
                result = [p for p in parsed if p["start_time"] > _grace_iso]
                if not result:
                    result = parsed  # fallback: return all if none are upcoming

    except Exception as e:
        print("LOOKUP S1 ERROR:", e)

    # Strategy 2: fetch all bookings and filter client-side by attendee email
    try:
        resp2 = _req2.get(
            "https://api.cal.com/v2/bookings",
            params={"take": 100},
            headers=headers,
        )
        print("LOOKUP S2 STATUS:", resp2.status_code)
        if resp2.status_code == 200:
            raw2 = resp2.json().get("data", [])
            print("LOOKUP S2 RAW COUNT:", len(raw2) if isinstance(raw2, list) else raw2)
            if isinstance(raw2, list):
                parsed2 = [_parse(b) for b in raw2
                           if _matches(_parse(b))
                           and b.get("status","").upper() not in ("CANCELLED", "REJECTED", "CANCELED")
                           and not b.get("rescheduled")]
                result2 = [p for p in parsed2 if p["start_time"] > now_iso]
                if not result2:
                    result2 = parsed2
                if result2:
                    return result2
    except Exception as e:
        print("LOOKUP S2 ERROR:", e)

    return []
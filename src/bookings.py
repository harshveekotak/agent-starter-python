AVAILABLE_SLOTS = [
    "10:00", "11:00", "12:00",
    "15:00", "16:00", "17:00"
]

def check_slots(preferred_time):
    if preferred_time in AVAILABLE_SLOTS:
        return True, [preferred_time]
    return False, AVAILABLE_SLOTS


import requests
import os

def create_cal_booking(start_utc, name, email, phone, timezone):
    url = "https://api.cal.com/v2/bookings"
    headers = {
        "Authorization": f"Bearer {os.getenv('CAL_API_KEY')}",
        "Content-Type": "application/json",
        "cal-api-version": "2024-08-13"
    }

    body = {
        "start": start_utc,
        "eventTypeSlug": os.getenv("CAL_EVENT_SLUG"),
        "username": os.getenv("CAL_USERNAME"),
        "attendee": {
            "name": name,
            "email": email,
            "phoneNumber": phone,
            "timeZone": timezone,
            "language": "en"
        }
    }

    r = requests.post(url, json=body, headers=headers)
    return r.json()

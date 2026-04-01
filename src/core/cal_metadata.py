import os
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env.local")

CAL_API_BASE = "https://api.cal.com/v2"

def fetch_events():
    api_key = os.getenv("CAL_API_KEY")

    if not api_key:
        raise RuntimeError("CAL_API_KEY not set")

    url = f"{CAL_API_BASE}/event-types?username=harshvee-kotak-79iy5y"

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    resp = requests.get(url, headers=headers)

    if resp.status_code != 200:
        print("FETCH EVENTS FAILED:", resp.text)
        return []

    data = resp.json()

    # Extract nested eventTypes correctly
    event_groups = data.get("data", {}).get("eventTypeGroups", [])

    events = []
    for group in event_groups:
        events.extend(group.get("eventTypes", []))

    return events


def resolve_events_for_services(selected_services: list[str],
    events: list[dict]):
    """
    Returns all Cal.com events that satisfy ALL selected services.
    """
    normalized = [
        s.lower().replace("_", " ")
        for s in selected_services
    ]

    matches = []
    for event in events:
        title = event.get("title", "").lower()

        if all(service in title for service in normalized):
            matches.append(event)

    return matches

def extract_services_from_combo(title: str):
    parts = [p.strip().lower() for p in title.split("+")]
    return parts
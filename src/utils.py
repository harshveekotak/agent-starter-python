from datetime import datetime
import pytz

def local_to_utc(date: str, time: str, timezone: str) -> str | None:
    """
    Convert local date & time to UTC ISO-8601 string (no microseconds).
    Example output: 2026-01-24T11:30:00Z
    Returns None on invalid input.
    """

    try:
        # Parse date + time
        local_tz = pytz.timezone(timezone)
        dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")

        # Localize to timezone
        dt = local_tz.localize(dt)

        dt = dt.replace(microsecond=0)

        # Convert to UTC and format
        return dt.astimezone(pytz.utc).isoformat().replace("+00:00", "Z")

    except Exception as e:
        print("UTC conversion error:", e)
        return None

def record_recommendation(session, key: str, rec_type: str, signals: dict):
    """
    Stores WHY something was recommended using structured signals.
    """
    session.recommendations[key] = {
        "type": rec_type,
        "signals": signals
    }

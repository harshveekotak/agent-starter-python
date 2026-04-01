from datetime import datetime, timedelta
import pytz


def is_weekend(date_str: str, timezone: str = "Asia/Kolkata") -> bool:
    tz = pytz.timezone(timezone)
    dt = tz.localize(datetime.strptime(date_str, "%Y-%m-%d"))
    return dt.weekday() >= 5


def next_weekday(date_str: str, timezone: str = "Asia/Kolkata") -> str:
    tz = pytz.timezone(timezone)
    dt = tz.localize(datetime.strptime(date_str, "%Y-%m-%d"))
    while dt.weekday() >= 5:
        dt += timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


def is_past_datetime(date_str: str, time_str: str, timezone: str) -> bool:
    """Returns True if the given date+time is in the past."""
    tz = pytz.timezone(timezone)

    if not date_str or not time_str:
        return False

    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        dt = tz.localize(dt)
    except Exception:
        return False
    return dt <= datetime.now(tz)


def minutes_since_start(start_iso: str, timezone: str = "Asia/Kolkata") -> float:
    """
    Returns minutes elapsed since start_iso (ISO string with Z).
    Negative = appointment is still in the future.
    Positive = appointment has already started.
    """
    tz = pytz.timezone(timezone)
    try:
        dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).astimezone(tz)
        return (datetime.now(tz) - dt).total_seconds() / 60
    except Exception:
        return 0


# Appointments started within this many minutes ago can still be cancelled (with a warning).
# Beyond this → hard block.
CANCEL_GRACE_MINUTES = 30
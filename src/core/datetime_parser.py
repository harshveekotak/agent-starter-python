import re
from datetime import datetime, timedelta
import pytz


_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_MONTH_PAT = r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"

# "11th march", "3rd of june", "march 11", "april 5th"
_DATE_RE = re.compile(
    rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_PAT})"
    rf"|({_MONTH_PAT})\s+(\d{{1,2}})(?:st|nd|rd|th)?",
    re.IGNORECASE,
)

# "3pm", "6:30 pm", "10 am"
_TIME_RE_AMPM = re.compile(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", re.IGNORECASE)

# "15:00", "13:30" (24-hour)
_TIME_RE_24 = re.compile(r"\b(\d{1,2}):(\d{2})\b")


_WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

def _parse_date_from_text(text_lower: str, now: datetime):
    """Extract a calendar date from free text. Returns a datetime or None."""

    import re as _re_guard
    
    # Guard: pure single digit with no month/weekday/date context
    # "7" alone → not a date. "7th", "march 7", "the 7th" → still valid
    _stripped = text_lower.strip()
    if _re_guard.fullmatch(r"\d{1,2}", _stripped):
        # It's ONLY a number like "7" or "12" with nothing else
        # This cannot be unambiguously a date without month context
        return None
    
    if "tomorrow" in text_lower:
        return now + timedelta(days=1)
    if "today" in text_lower:
        return now

    # Weekday detection: "friday", "next friday", "this coming friday", etc.
    # Check for weekday names before the month-day pattern
    _next_hint = "next" in text_lower or "coming" in text_lower or "this" in text_lower
    for _wname, _wnum in _WEEKDAYS.items():
        # Match whole word only
        if re.search(r"\b" + _wname + r"\b", text_lower):
            days_ahead = (_wnum - now.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7  # same weekday = next week
            elif _next_hint and days_ahead < 7:
                days_ahead += 7 if "next" in text_lower else 0
            return now + timedelta(days=days_ahead)

    m = _DATE_RE.search(text_lower)
    if m:
        day   = m.group(1) or m.group(4)
        month_s = m.group(2) or m.group(3)
        month_key = month_s[:3].lower()
        month_num = _MONTHS.get(month_s.lower()) or _MONTHS.get(month_key)
        if not month_num:
            return None
        day = int(day)
        year = now.year
        try:
            candidate = now.replace(year=year, month=month_num, day=day,
                                    hour=0, minute=0, second=0, microsecond=0)
        except ValueError:
            return None
        if candidate.date() < now.date():
            try:
                candidate = candidate.replace(year=year + 1)
            except ValueError:
                pass
        return candidate

    # Fallback: dateparser on full text
    # Only used when weekday/month-day patterns didn't match
    try:
        import dateparser
        result = dateparser.parse(
            text_lower,
            settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": now,
                      "TIMEZONE": "Asia/Kolkata", "PREFER_DAY_OF_MONTH": "first"},
        )
        # Sanity check: reject dates more than 2 years out
        if result and abs((result.year - now.year)) > 1:
            return None
        return result
    except Exception:
        return None


# Bare "at 11", "at 3", "for 11", "for 3" -- no am/pm
_TIME_RE_BARE = re.compile(r"(?:at|for|@)\s+(\d{1,2})(?:\s|$|\?|,|\.)", re.IGNORECASE)

def _parse_time_from_text(text_lower: str):
    """Extract (hour, minute) from free text."""
    m = _TIME_RE_AMPM.search(text_lower)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        meridian = m.group(3).lower()
        if meridian == "pm" and hour != 12:
            hour += 12
        if meridian == "am" and hour == 12:
            hour = 0
        return hour, minute
    m = _TIME_RE_24.search(text_lower)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mn <= 59:
            return h, mn
    # Bare hour: "at 3", "for 11" -- no AM/PM given
    # Salon context: assume PM for hours 1-7 (nobody books a facial at 3am)
    # Hours 8-12 left as-is (8am open, 9am open, etc.)
    m = _TIME_RE_BARE.search(text_lower)
    if m:
        h = int(m.group(1))
        if 1 <= h <= 7:
            return h + 12, 0  # 3 -> 15:00, 6 -> 18:00, etc.
        if 8 <= h <= 12:
            return h, 0  # 10 -> 10:00 AM (reasonable salon open hour)
    return None, None


def extract_datetime(text: str):
    """
    Extract (date_str, time_str) from natural language text.
    Returns (None, None) if either component is missing.
    date_str: "YYYY-MM-DD"  |  time_str: "HH:MM"
    """
    tz = pytz.timezone("Asia/Kolkata")
    now = datetime.now(tz)
    text_lower = text.lower()

    date = _parse_date_from_text(text_lower, now)
    hour, minute = _parse_time_from_text(text_lower)

    # If neither found, return nothing
    if date is None and hour is None:
        return None, None

    # If only time found (e.g. "3pm"), return (None, time)
    if date is None:
        h, m = hour, minute
        try:
            result_time = now.replace(hour=h, minute=m or 0, second=0, microsecond=0)
            return None, result_time.strftime("%H:%M")
        except Exception:
            return None, None

    # If only date found (e.g. "tomorrow"), return (date, None)
    if hour is None:
        return date.strftime("%Y-%m-%d"), None

    try:
        if hasattr(date, "tzinfo") and date.tzinfo:
            result = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
        else:
            result = tz.localize(date.replace(hour=hour, minute=minute, second=0, microsecond=0))
    except Exception:
        result = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    return result.strftime("%Y-%m-%d"), result.strftime("%H:%M")
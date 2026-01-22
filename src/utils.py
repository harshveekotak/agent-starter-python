from datetime import datetime
import pytz

def local_to_utc(date_str, time_str, timezone):
    """
    date_str: YYYY-MM-DD
    time_str: HH:MM (24h)
    timezone: Asia/Kolkata
    """
    tz = pytz.timezone(timezone)
    local_dt = tz.localize(
        datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    )
    return local_dt.astimezone(pytz.utc).isoformat().replace("+00:00", "Z")

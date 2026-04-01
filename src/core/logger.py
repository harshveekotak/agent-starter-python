import json
from datetime import datetime, timezone

LOG_FILE = "conversation_logs.json"

def log_event(event_type: str, payload: dict):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        "data": payload
    }

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print("Logging failed:", e)
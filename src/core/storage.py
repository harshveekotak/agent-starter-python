import json
import os

from pathlib import Path
_BASE = Path(__file__).resolve().parent.parent  # → src/
_DATA_DIR = _BASE / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_FILE = str(_DATA_DIR / "sessions.json")
PROFILES_FILE = str(_DATA_DIR / "profiles.json")


# -- Session (mid-flow booking state) -----------------------------------------

def load_latest_session() -> dict | None:
    """Load the current mid-flow session, or None if none exists."""
    print(f"DEBUG LOAD FROM: {SESSIONS_FILE}")
    
    if not os.path.exists(SESSIONS_FILE):
        return None
    try:
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data[-1] if data else None
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def save_session(session_dict: dict):
    """Overwrite sessions.json with the current session state (single object)."""
    try:
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(session_dict, f, indent=2)
    except Exception as e:
        print(f"save_session error: {e}")


def clear_session():
    """Delete the session file -- called after a booking completes or is cancelled."""
    if os.path.exists(SESSIONS_FILE):
        try:
            os.remove(SESSIONS_FILE)
        except Exception as e:
            print(f"clear_session error: {e}")


# -- Profile (persists across sessions: identity + last booking ref) -----------

def load_profile(email: str = None) -> dict:
    """Load profile for a specific email, or empty dict if not found."""
    if not email:
        return {}
    
    if not os.path.exists(PROFILES_FILE):
        return {}
    
    try:
        with open(PROFILES_FILE, "r", encoding="utf-8") as f:
            all_profiles = json.load(f)  # {"alice@ex.com": {...}, "bob@ex.com": {...}}
        return all_profiles.get(email, {})
    except Exception:
        return {}


def save_profile(
    email: str,
    name: str = None,
    phone: str = None,
    booking_id: int = None,
    booking_uid: str = None,
    booking_date: str = None,
    booking_time: str = None,
    items_selected: list = None,
):
    
    """Save profile for a specific email."""
    if not email:
        print("! save_profile called without email — ignoring")
        return
    
    # Load all profiles
    all_profiles = {}
    if os.path.exists(PROFILES_FILE):
        try:
            with open(PROFILES_FILE, "r", encoding="utf-8") as f:
                all_profiles = json.load(f)
        except Exception:
            pass
    
    # Update this user's profile
    if email not in all_profiles:
        all_profiles[email] = {}

    
    profile = all_profiles[email]
    profile["email"] = email
    if name is not None:           profile["name"]           = name
    if phone is not None:          profile["phone"]          = phone
    if booking_id is not None:     profile["booking_id"]     = booking_id
    if booking_uid is not None:    profile["booking_uid"]    = booking_uid
    if booking_date is not None:   profile["booking_date"]   = booking_date
    if booking_time is not None:   profile["booking_time"]   = booking_time
    if items_selected is not None: profile["items_selected"] = items_selected

    try:
        with open(PROFILES_FILE, "w", encoding="utf-8") as f:
            json.dump(all_profiles, f, indent=2)
    except Exception as e:
        print(f"save_profile error: {e}")


def clear_profile(email: str = None):
    """Deletes profile for a specific email, or entire file if email=None."""
    if not email:
        if os.path.exists(PROFILES_FILE):
            try:
                os.remove(PROFILES_FILE)
            except Exception as e:
                print(f"clear_profile error: {e}")
        return

    if not os.path.exists(PROFILES_FILE):
        return

    try:
        with open(PROFILES_FILE, "r", encoding="utf-8") as f:
            all_profiles = json.load(f)

        all_profiles.pop(email, None)  # Remove this user
        
        with open(PROFILES_FILE, "w", encoding="utf-8") as f:
            json.dump(all_profiles, f, indent=2)

    except Exception as e:
        print(f"clear_profile error: {e}")

from core.brain import Brain

_brain_instance = None

def _get_brain():
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = Brain()
    return _brain_instance

brain_instance = property(_get_brain)  # won't work as module-level

def handle_user_message(text: str):
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = Brain()
    return _brain_instance.handle_text(text)

def get_selected_services():
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = Brain()
    return _brain_instance.session.items_selected

def get_booking_details():
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = Brain()
    return {
        "services": _brain_instance.session.items_selected,
        "date": _brain_instance.session.date,
        "time": _brain_instance.session.time,
        "name": _brain_instance.session.name,
        "email": _brain_instance.session.email,
        "phone": _brain_instance.session.phone,
        "ready": _brain_instance.session.ready_to_book,
    }
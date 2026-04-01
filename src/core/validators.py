import re

def is_valid_email(email: str) -> bool:
    if not email:
        return False

    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(email_pattern, email) is not None


def is_valid_phone(phone: str) -> bool:
    if not phone:
        return False

    # Accept only last 10 digits (India-friendly)
    digits = "".join(filter(str.isdigit, phone))
    return len(digits) == 10

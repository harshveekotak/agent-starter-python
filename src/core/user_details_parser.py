import re

# ----------------------------
# USER DETAILS EXTRACTION
# ----------------------------

def extract_user_details(text: str):
    """
    Extract name, email, phone from free text.

    Example:
    "My name is Harshvee, email harshvee@gmail.com, phone 9822334455"

    Works with:
    - "Harshvee, honey@gmail.com"
    - "Name: Harshvee, Email: honey@gmail.com"
    - "Harshvee honey@gmail.com"
    - "Harshvee honey@gmail.com 9822334455"
    - "Harshvee honey@gmail.com +91 9822334455"
    - "Harshvee honey@gmail.com 91-9822334455"
    - "Harshvee honey@gmail.com +91-9822334455"
    - "Harshvee honey@gmail.com +919822334455"
    - "Harshvee honey@gmail.com +919822334455"

    Returns:
        (name, email, phone)
    """

    name = extract_name(text)
    email = extract_email(text)
    phone = extract_phone(text)

    return name, email, phone


# ----------------------------
# EMAIL EXTRACTION
# ----------------------------

def extract_email(text: str):
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match = re.search(email_pattern, text)

    if match:
        return match.group()

    return None


# ----------------------------
# PHONE EXTRACTION
# ----------------------------

def extract_phone(text: str):
    """
    Supports:
    - 9822334455
    - +91 9822334455
    - 91-9822334455
    """

    phone_pattern = r"(\+?\d{1,3}[\s-]?)?\d{10}"
    match = re.search(phone_pattern, text)

    if match:
        phone = match.group()

        # Remove spaces/dashes
        phone = phone.replace(" ", "").replace("-", "")

        return phone

    return None


# ----------------------------
# NAME EXTRACTION
# ----------------------------

def extract_name(text: str):
    """
    Flexible name extraction.

    Works with:
    - "My name is Harshvee"
    - "I am Harshvee"
    - "Harshvee here"
    - "Name: Harshvee"
    - "Harshvee, honey@gmail.com"
    - "Harshvee honey@gmail.com 9822334455"
    - "no use harshvee as name and email is harshvee@gmail.com"
    - "use harshvee as name, harshvee@gmail.com",
    - "honey is my name and my mail is honeykotak5@gmail.com"
    """

    SKIP_WORDS = {
        "name", "email", "is", "my", "the", "a", "an", "and", "with",
        "for", "at", "on", "of", "or", "to", "by", "in", "just", "use",
        "please", "want", "need", "book", "schedule", "tomorrow", "today",
        "evening", "morning", "afternoon", "no", "yes", "not", "as", "but",
        "so", "if", "do", "it", "me", "us", "we", "he", "she", "they",
        "that", "this", "its", "be", "am", "are", "was", "were", "will",
        "can", "could", "would", "should", "have", "has", "had", "different",
        "other", "instead", "nope", "nah", "okay", "ok", "sure", "right",
        "actually", "said", "i", "here", "mail",
    }

    text_lower = text.lower().strip()

    # Pattern-based matches -- most specific first
    name_patterns = [
        r"([A-Za-z]+) is my name",
        r"my name is ([A-Za-z]+)",
        r"name is ([A-Za-z]+)",
        r"name:\s*([A-Za-z]+)",
        r"\bname\s+([A-Za-z]+)",
        r"i am ([A-Za-z]+)",
        r"this is ([A-Za-z]+)",
        r"call me ([A-Za-z]+)",
        r"use ([A-Za-z]+) as (?:my )?name",
        r"use ([A-Za-z]+),",
        r"([A-Za-z]+) as (?:the |my )?name",
    ]

    for pattern in name_patterns:
        match = re.search(pattern, text_lower)
        if match:
            candidate = match.group(1).capitalize()
            if candidate.lower() not in SKIP_WORDS:
                return candidate

    # Fallback: extract from text before email
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    if email_match:
        before_email = text[:email_match.start()].strip()
        before_email = re.sub(
            r'\b(name|email|is|my|use|as|no|the|and|with|for|at|of|or|to|by|in|said|i|mail)\b',
            '', before_email, flags=re.IGNORECASE
        )
        before_email = before_email.replace(",", "").strip()
        words = [w for w in before_email.split()
                 if w.lower() not in SKIP_WORDS and w.isalpha()]
        if words:
            return words[0].capitalize() 

    # Fallback 2: before phone
    phone_match = re.search(r"\d{10}", text)
    if phone_match:
        before_phone = text[:phone_match.start()].strip()
        words = [w for w in before_phone.split()
                 if w.lower() not in SKIP_WORDS and w.isalpha()]
        if words:
            return words[0].capitalize()

    return None
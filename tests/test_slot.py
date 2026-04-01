import sys
sys.path.insert(0, "src")
import re

# ── PASTE FUNCTION INLINE FOR TESTING ──
def extract_name_test(text: str):

    SKIP_WORDS = {
        "name", "email", "is", "my", "the", "a", "an", "and", "with",
        "for", "at", "on", "of", "or", "to", "by", "in", "just", "use",
        "please", "want", "need", "book", "schedule", "tomorrow", "today",
        "evening", "morning", "afternoon", "no", "yes", "not", "as", "but",
        "so", "if", "do", "it", "me", "us", "we", "he", "she", "they",
        "that", "this", "its", "be", "am", "are", "was", "were", "will",
        "can", "could", "would", "should", "have", "has", "had", "different",
        "other", "instead", "nope", "nah", "okay", "ok", "sure", "right",
        "actually", "said", "i", "here",
    }

    text_lower = text.lower().strip()

    name_patterns = [
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
            r'\b(name|email|is|my|use|as|no|the|and|with|for|at|of|or|to|by|in|said|i)\b',
            '', before_email, flags=re.IGNORECASE
        )
        before_email = before_email.replace(",", "").strip()
        words = [w for w in before_email.split()
                 if w.lower() not in SKIP_WORDS and w.isalpha()]
        if words:
            return words[-1].capitalize()

    # Fallback 2: before phone
    phone_match = re.search(r"\d{10}", text)
    if phone_match:
        before_phone = text[:phone_match.start()].strip()
        words = [w for w in before_phone.split()
                 if w.lower() not in SKIP_WORDS and w.isalpha()]
        if words:
            return words[-1].capitalize()

    return None


# ── TESTS ──
tests = [
    ("no use harshvee as name and email is harshveekotak@gmail.com", "Harshvee"),
    ("no i said to use harshvee as name and harshveekotak@gmail.com as email", "Harshvee"),
    ("honey, honeykotak5@gmail.com", "Honey"),
    ("my name is harshvee", "Harshvee"),
    ("use harshvee, harshveekotak@gmail.com", "Harshvee"),
    ("harshvee, harshveekotak@gmail.com", "Harshvee"),
    ("it's harshvee and harshveekotak@gmail.com", "Harshvee"),
    ("harshvee harshveekotak@gmail.com", "Harshvee"),
    ("honey honeykotak5@gmail.com", "Honey"),
    ("harshvee harshveekotak@gmail.com 9822334455", "Harshvee"),

]

print("--- Name Extraction Tests ---")
all_passed = True
for text, expected in tests:
    result = extract_name_test(text)
    status = "✅" if result == expected else "❌"
    if result != expected:
        all_passed = False
    print(f"{status} Input:    {text}")
    print(f"   Expected: {expected} | Got: {result}")
    print()

print("ALL PASSED ✅" if all_passed else "SOME FAILED ❌")
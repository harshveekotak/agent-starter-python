# Service names MUST exactly match Cal.com MultiSelect option values
# Only salon domain is active — other domains removed to prevent false matches
CATALOG = {
    "salon": [
        "Haircut", "Hair Spa", "Massage",
        "Facial", "Manicure", "Pedicure"
    ],
}

# Combo suggestions: if user books service X, suggest adding Y
COMBO_SUGGESTIONS = {
    "Haircut":  ["Hair Spa", "Massage"],
    "Hair Spa": ["Haircut", "Facial"],
    "Massage":  ["Facial", "Haircut"],
    "Facial":   ["Massage", "Manicure"],
    "Manicure": ["Pedicure", "Facial"],
    "Pedicure": ["Manicure", "Massage"],
}
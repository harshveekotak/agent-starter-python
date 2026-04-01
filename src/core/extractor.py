from sentence_transformers import SentenceTransformer, util
from core.embeddings import embed_text

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Cal.com exact option values
CAL_SERVICES = ["Haircut", "Hair Spa", "Massage", "Facial", "Manicure", "Pedicure"]

# ============================================================================
# Rich semantic descriptions — enables natural language understanding
# No hardcoded aliases needed
# ============================================================================
SERVICE_DESCRIPTIONS = {
    "Haircut": (
        "haircut hair cut trim trimming hair trim cut my hair "
        "hair styling hairstyle blow dry blowdry blowout "
        "shave beard trim clipper cut buzz cut hair shaping "
        "layers bangs fringe hair design taper fade undercut "
        "bob pixie mullet crew cut pompadour quiff"
    ),
    "Hair Spa": (
        "hair spa hair treatment hair therapy spa for hair "
        "deep conditioning conditioning treatment hair mask "
        "keratin treatment protein treatment hair detox "
        "scalp treatment scalp massage hair rejuvenation "
        "hair smoothening hair repair nourishing treatment "
        "hair relaxation hair hydration hair revitalization"
    ),
    "Massage": (
        "massage body massage full body massage back massage "
        "shoulder massage head massage scalp massage "
        "relaxation massage therapeutic massage deep tissue "
        "aromatherapy stress relief tension release "
        "body relaxation spa massage wellness massage "
        "neck massage foot massage hand massage chair massage"
    ),
    "Facial": (
        "facial face treatment facial treatment face cleanup "
        "clean up face clean skin care skincare skin treatment "
        "face mask face therapy anti-aging facial "
        "hydrating facial glow facial brightening facial "
        "acne treatment pore cleansing exfoliation face scrub "
        "face polish face glow radiance facial detox facial"
    ),
    "Manicure": (
        "manicure mani hand nails nail care nail treatment "
        "fingernails nail polish nail art gel nails "
        "nail shaping cuticle care hand care nail spa "
        "hand treatment nail filing nail buffing hand massage "
        "acrylic nails shellac nails nail extension hand grooming"
    ),
    "Pedicure": (
        "pedicure pedi foot nails feet nails toe nails "
        "foot care foot treatment foot spa foot massage "
        "feet care toe care foot scrub heel treatment "
        "callus removal foot exfoliation foot relaxation "
        "toe polish foot grooming feet grooming foot soak"
    ),
}


def extract_service(user_text: str, full_catalog: dict):
    """
    Extracts a single service from user text using semantic similarity.
    Now uses rich descriptions instead of just service names.
    
    Examples:
        "I want a trim" → ("salon", "Haircut")
        "buzz cut please" → ("salon", "Haircut")
        "foot spa" → ("salon", "Pedicure")
        "glow treatment" → ("salon", "Facial")
    """
    # Pre-check: if text looks like a comma-separated list, score each segment
    segments = [s.strip() for s in user_text.replace(";", ",").split(",") if s.strip()]
    if len(segments) > 1:
        best_match = None
        best_domain = None
        best_score = 0
        for seg in segments:
            r = _score_text_semantic(seg)
            if r and r[2] > best_score:
                best_domain, best_match, best_score = r
        print(f"DEBUG Match: {best_domain}:{best_match} ({best_score:.2f})")
        if best_score >= 0.25:  # Lower threshold for semantic matching
            return best_domain, best_match
        return None

    return _single_extract_semantic(user_text)


def extract_all_services(user_text: str, full_catalog: dict) -> list:
    """
    Returns ALL services mentioned in the text using semantic matching.
    
    Examples:
        "haircut and massage" → ["Haircut", "Massage"]
        "mani pedi facial" → ["Manicure", "Pedicure", "Facial"]
        "trim with foot spa" → ["Haircut", "Pedicure"]
    """
    found = []
    seen = set()

    # 1. Split on common delimiters
    import re
    segments = re.split(r',|;|\band\b|\bwith\b|\bplus\b|\balso\b', user_text.lower())
    segments = [s.strip() for s in segments if s.strip()]
    
    if len(segments) > 1:
        for seg in segments:
            r = _score_text_semantic(seg)
            if r and r[2] >= 0.25 and r[1] not in seen:
                found.append(r[1])
                seen.add(r[1])
        if found:
            return found

    # 2. Keyword scan for explicit service names (fast path)
    text_l = user_text.lower()
    for svc in CAL_SERVICES:
        if svc.lower() in text_l and svc not in seen:
            found.append(svc)
            seen.add(svc)
    if found:
        return found

    # 3. Semantic single extract as fallback
    r = _single_extract_semantic(user_text)
    if r and r[1] not in seen:
        found.append(r[1])
    
    return found


def _score_text_semantic(text: str):
    """
    Score text against rich service descriptions using embeddings.
    Returns (domain, service, score) or None.
    
    This is the core LLM-powered matcher — no hardcoded aliases.
    """
    try:
        user_vec = embed_text(text.lower())
        
        best_match = None
        best_score = 0.0
        
        for service in CAL_SERVICES:
            desc = SERVICE_DESCRIPTIONS[service]
            desc_vec = embed_text(desc)
            score = float(util.cos_sim(user_vec, desc_vec))
            
            if score > best_score:
                best_score = score
                best_match = service
        
        if best_match and best_score >= 0.20:  # Lowered threshold for better recall
            return ("salon", best_match, best_score)
        
        return None
    
    except Exception as e:
        print(f"DEBUG SEMANTIC MATCH ERROR: {e}")
        return None


def _single_extract_semantic(user_text: str):
    """
    Single-service extraction using semantic descriptions.
    Returns (domain, service) or None.
    """
    result = _score_text_semantic(user_text)
    if result:
        domain, service, score = result
        print(f"DEBUG Match: {domain}:{service} ({score:.2f})")
        if score >= 0.20:
            return domain, service
    return None
from core.catalog import CATALOG, COMBO_SUGGESTIONS
from utils import record_recommendation


def recommend(session, domain: str, selected: list[str]):
    """
    Suggests the most relevant unselected service based on what's already selected.
    Uses COMBO_SUGGESTIONS for intelligent pairing, falls back to catalog order.
    """
    if domain is None:
        return None

    # Try to find a pairing suggestion based on selected services
    for service in selected:
        candidates = COMBO_SUGGESTIONS.get(service, [])
        for candidate in candidates:
            if candidate not in selected and candidate in CATALOG.get(domain, []):
                record_recommendation(
                    session,
                    key=candidate,
                    rec_type="service",
                    signals={
                        "domain": domain,
                        "triggered_by": service,
                        "already_selected": selected,
                        "strategy": "combo_pairing"
                    }
                )
                return candidate

    # Fallback: first unselected service in catalog
    for service in CATALOG.get(domain, []):
        if service not in selected:
            record_recommendation(
                session,
                key=service,
                rec_type="service",
                signals={
                    "domain": domain,
                    "already_selected": selected,
                    "strategy": "catalog_order"
                }
            )
            return service

    return None
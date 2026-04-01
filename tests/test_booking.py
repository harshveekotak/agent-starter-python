import pytest
from core.sessions import Session
from core.time_utils import is_weekend, next_weekday, is_past_datetime

def test_is_weekend():
    assert is_weekend("2026-02-14") is True   # Saturday
    assert is_weekend("2026-02-16") is False  # Monday

def test_next_weekday():
    assert next_weekday("2026-02-14") == "2026-02-16"

def test_session_object():
    session = Session(domain="booking")
    assert session.domain == "booking"
    assert session.items_selected == []
    assert session.completed is False

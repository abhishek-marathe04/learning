"""Run every page through Streamlit's AppTest to catch runtime errors."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGES = [str(ROOT / "pages" / p) for p in (
    "1_scoreboard.py", "2_downhill.py", "3_backprop.py",
    "4_bends.py", "5_next_letter.py",
)]


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page):
    at = AppTest.from_file(page, default_timeout=90).run()
    assert not at.exception, f"{page} raised: {at.exception}"

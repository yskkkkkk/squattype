from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_app_starts():
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(app_path).run(timeout=60)
    assert not at.exception

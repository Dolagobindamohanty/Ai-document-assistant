from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_ux_smoke_streamlit_app_loads():
    """UX-SMOKE: Verify that app.py compiles, initializes, and mounts all root elements without error."""
    app_path = Path(__file__).resolve().parent.parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    # Verify no unhandled exceptions during run
    assert not at.exception

    # Verify main title and widgets are mounted
    assert len(at.title) >= 1
    assert "AI Document Assistant" in at.title[0].value
    assert len(at.selectbox) >= 1
    assert at.selectbox[0].label == "Choose a tool"

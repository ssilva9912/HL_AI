import importlib.util
import tomllib
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]


def load_theme():
    spec = importlib.util.spec_from_file_location("theme", ROOT / "frontend" / "theme.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


theme = load_theme()


def test_status_pill_shows_whether_the_backend_is_reachable() -> None:
    assert "Online" in theme.status_pill(True)
    assert "off" not in theme.status_pill(True)
    assert "Offline" in theme.status_pill(False)
    assert 'class="hl-pill off"' in theme.status_pill(False)


def test_header_carries_the_name_tagline_and_status() -> None:
    html = theme.header_html(online=False)

    assert "Homelab AI" in html
    assert theme.TAGLINE in html
    assert "Offline" in html


def test_welcome_card_offers_shortcuts_to_real_pages() -> None:
    assert "how can I help" in theme.welcome_html()
    assert {target for _, _, target in theme.QUICK_ACTIONS} == {"documents", "forms", "metrics"}
    assert all(label and icon.startswith(":material/") for label, icon, _ in theme.QUICK_ACTIONS)


def test_styles_are_injected_as_trusted_html() -> None:
    streamlit = MagicMock()
    theme.st = streamlit

    theme.inject_styles()

    markup = streamlit.markdown.call_args.args[0]
    assert markup.lstrip().startswith("<style>")
    assert streamlit.markdown.call_args.kwargs == {"unsafe_allow_html": True}


def test_streamlit_theme_defines_light_and_dark_palettes() -> None:
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))

    assert config["client"]["toolbarMode"] == "minimal"
    assert {"light", "dark"} <= set(config["theme"])
    assert config["theme"]["dark"]["backgroundColor"] != config["theme"]["light"]["backgroundColor"]


def test_the_container_image_ships_the_streamlit_config() -> None:
    assert "COPY .streamlit" in (ROOT / "Dockerfile").read_text(encoding="utf-8")

import importlib.util
from pathlib import Path
from types import SimpleNamespace

path = Path(__file__).resolve().parents[1] / "frontend" / "forms_panel.py"
spec = importlib.util.spec_from_file_location("forms_panel", path)
assert spec is not None and spec.loader is not None
panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel)


def field(
    field_id: str,
    label: str,
    *,
    value: str | None = None,
    verified: bool = False,
    source: str | None = None,
    kind: str = "text",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=field_id, label=label, value=value, verified=verified, source=source, kind=kind
    )


FIELDS = [
    field("a", "Name", value="Alex", verified=True, source="profile.txt"),
    field("b", "Phone", value="555", verified=False),
    field("c", "Salary"),
    field("d", "Newsletter", kind="checkbox"),
]


def test_check_labels_say_how_far_to_trust_a_value() -> None:
    assert panel.check_label("Alex", True, "profile.txt") == "Found in profile.txt"
    assert panel.check_label("Alex", True, None) == "Found in your documents"
    assert "please check" in panel.check_label("555", False, None)
    assert panel.check_label(None, False, None) == "Not found in your documents"
    assert panel.check_label("", True, "profile.txt") == "Not found in your documents"


def test_rows_cover_text_fields_only() -> None:
    assert panel.build_rows(FIELDS) == [
        {"Field": "Name", "Value": "Alex", "Check": "Found in profile.txt"},
        {
            "Field": "Phone",
            "Value": "555",
            "Check": "Not word for word in your documents, please check",
        },
        {"Field": "Salary", "Value": "", "Check": "Not found in your documents"},
    ]


def test_edited_values_and_ticked_boxes_become_the_fill_values() -> None:
    rows = [
        {"Field": "Name", "Value": "  Alex Rivera ", "Check": ""},
        {"Field": "Phone", "Value": "", "Check": ""},
        {"Field": "Salary", "Value": "90k", "Check": ""},
    ]

    values = panel.collect_values(FIELDS, rows, {"d": True, "e": False})

    assert values == {"a": "Alex Rivera", "c": "90k", "d": "yes"}


def test_summary_counts_suggestions_and_unverified_ones() -> None:
    assert panel.summarize(FIELDS) == (
        "Found suggestions for 2 of 3 fields. 1 could not be matched word for word, so check those."
    )
    assert panel.summarize([field("a", "Name", value="Alex", verified=True)]) == (
        "Found suggestions for 1 of 1 fields."
    )

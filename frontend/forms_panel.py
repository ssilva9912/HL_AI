from typing import Any

CHECKED_VALUE = "yes"


def check_label(value: str | None, verified: bool, source: str | None) -> str:
    """How trustworthy a suggested value is, in a few words for the review table."""

    if not value:
        return "Not found in your documents"
    if verified:
        return f"Found in {source}" if source else "Found in your documents"
    return "Not word for word in your documents, please check"


def build_rows(fields: list[Any]) -> list[dict[str, str]]:
    """One editable row per text field; checkboxes are ticked separately."""

    return [
        {
            "Field": field.label,
            "Value": field.value or "",
            "Check": check_label(field.value, field.verified, field.source),
        }
        for field in fields
        if field.kind == "text"
    ]


def collect_values(
    fields: list[Any],
    rows: list[dict[str, str]],
    ticked: dict[str, bool],
) -> dict[str, str]:
    """The field id to value mapping to write into the form, skipping blanks."""

    text_fields = [field for field in fields if field.kind == "text"]
    values = {
        field.id: str(row.get("Value") or "").strip()
        for field, row in zip(text_fields, rows, strict=True)
    }
    values.update({field_id: CHECKED_VALUE for field_id, on in ticked.items() if on})
    return {field_id: value for field_id, value in values.items() if value}


def summarize(fields: list[Any]) -> str:
    text_fields = [field for field in fields if field.kind == "text"]
    found = sum(1 for field in text_fields if field.value)
    unverified = sum(1 for field in text_fields if field.value and not field.verified)

    summary = f"Found suggestions for {found} of {len(text_fields)} fields."
    if unverified:
        summary += f" {unverified} could not be matched word for word, so check those."
    return summary

def normalize_name(name: str) -> str:
    """Normalize registry keys and user-provided identifiers."""
    return name.strip().lower()
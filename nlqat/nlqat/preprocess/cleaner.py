
def clean_text(text: str) -> str:
    """
    Cleans the input text by removing extra whitespace and stripping.
    """
    if not text:
        return ""
    text = text.strip()
    return " ".join(text.split())

import re


def normalize_whitespace(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_urls(text: str) -> str:
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\(?bit\.ly/\w+\)?", "", text)
    return text


def remove_social_artifacts(text: str) -> str:
    text = str(text)
    text = re.sub(r"@\w+\s*:\s*-\s*", "", text)
    text = re.sub(r"\[\d{4}\s*[A-Z]{3}\]\s*-?\s*", "", text)
    return text


def remove_source_footer(text: str) -> str:
    text = str(text)
    text = re.sub(r"--\s*Source link:.*$", "", text, flags=re.IGNORECASE)
    return text


def clean_text_basic(text: str) -> str:
    text = str(text)
    text = remove_urls(text)
    text = remove_social_artifacts(text)
    text = remove_source_footer(text)
    text = normalize_whitespace(text)
    return text


def clean_article_text(text: str) -> str:
    return clean_text_basic(text)


def clean_claim_text(text: str) -> str:
    return clean_text_basic(text)
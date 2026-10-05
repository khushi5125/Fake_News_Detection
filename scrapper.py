import re
import requests
import trafilatura
from bs4 import BeautifulSoup


def clean_text(text: str) -> str:
    """Normalize extra whitespaces and line breaks."""
    return re.sub(r"\s+", " ", text).strip()


def scrape_article(url: str) -> dict:
    """Fetch article from URL, returning title, body, and the evaluate.py-compatible input_text."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/118.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html = response.text
    except Exception as exc:
        return {"error": f"Failed to fetch URL: {str(exc)}"}

    # Attempt primary extraction with trafilatura
    extracted_text = trafilatura.extract(html, include_comments=False)
    title = ""

    # Extract title with BeautifulSoup
    try:
        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text().strip()
    except Exception:
        title = ""

    # Fallback if trafilatura fails to grab text
    if not extracted_text:
        try:
            paragraphs = soup.find_all("p")
            extracted_text = " ".join([p.get_text() for p in paragraphs])
        except Exception:
            extracted_text = ""

    title = clean_text(title)
    extracted_text = clean_text(extracted_text or "")

    # Matching the exact input_text preprocessing from evaluate.py:
    # If text exists: f"{title}. {text}"
    # Otherwise fallback strictly to title
    if extracted_text and title:
        combined_input = f"{title}. {extracted_text}"
    elif extracted_text:
        combined_input = extracted_text
    elif title:
        combined_input = title
    else:
        return {"error": "Could not extract readable title or text from URL."}

    return {
        "title": title,
        "text": extracted_text,
        "input_text": combined_input,
    }
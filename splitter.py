import logging

import html2text
import spacy
from bs4 import BeautifulSoup

from edgar import edgar_file

logger = logging.getLogger(__name__)

DEFAULT_TEXT_CHUNK_SIZE = 3500


def chunk_text(
    content: str, chunk_size: int = DEFAULT_TEXT_CHUNK_SIZE, method: str = "spacy"
) -> list[str]:
    if method == "spacy":
        return chunk_text_by_spacy(content, chunk_size)
    else:
        raise RuntimeError(f"Unknown method: {method}")


def chunk_text_by_spacy(content: str, chunk_size: int) -> list[str]:
    """
    Split a text into chunks of size chunk_size

    Args:
        text (str): The text to split into chunks
        chunk_size (int): The size of each chunk

    Returns:
        list[str]: A list of text chunks
    """
    # Load SpaCy NLP model
    nlp = spacy.load("en_core_web_sm")
    logger.debug("chunk_text: loaded model")

    # Split content into paragraphs (based on double newline)
    paragraphs = content.split("\n\n")

    chunks = []
    current_chunk = []
    current_size = 0

    for paragraph in paragraphs:
        # Check if the paragraph contains a Markdown table
        if paragraph.strip().startswith("|") and paragraph.strip().endswith("|"):
            # Treat Markdown tables as single units
            paragraph_size = len(paragraph)
            if current_size + paragraph_size > chunk_size:
                # Save current chunk and start a new one
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [paragraph]
                current_size = paragraph_size
            else:
                current_chunk.append(paragraph)
                current_size += paragraph_size
        else:
            # Process paragraph using SpaCy for sentence tokenization
            doc = nlp(paragraph)
            sentences = [sent.text for sent in doc.sents]
            for sentence in sentences:
                sentence_size = len(sentence)
                if current_size + sentence_size > chunk_size:
                    # Save current chunk and start a new one
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [sentence]
                    current_size = sentence_size
                else:
                    current_chunk.append(sentence)
                    current_size += sentence_size

    # Add any remaining content
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def trim_html_content(filing_html_path: str) -> str:
    """
    remove the hidden div and convert the rest of html into text
    """
    content = edgar_file(filing_html_path)
    if not content:
        return ""

    soup = BeautifulSoup(content, "html.parser")

    style_lambda = lambda value: value and "display:none" in value.replace(" ", "")  # noqa
    div_to_remove = soup.find("div", style=style_lambda)

    if div_to_remove:
        div_to_remove.decompose()  # type: ignore

    return default_text_converter().handle(str(soup))


def default_text_converter():
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = True
    converter.ignore_emphasis = True
    converter.body_width = 0
    return converter

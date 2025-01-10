import logging
from datetime import datetime

import html2text
import spacy
from bs4 import BeautifulSoup

from .datastore import save_chunks
from .edgar import SECFiling

logger = logging.getLogger(__name__)

DEFAULT_TEXT_CHUNK_SIZE = 3500


def chunk_filing(
    filing: SECFiling,
    form_type: str,
    method: str = "spacy",
    tags: list[str] = [],
    table_name: str = "",  # leave empty if dryrun
) -> tuple[int, list[str]]:
    logger.debug(f"chunk_filing form {form_type} of {filing}")

    if filing:
        filing_path, filing_content = filing.get_doc_content(form_type, max_items=1)[0]

        if not filing_path.endswith(".html") and not filing_path.endswith(".htm"):
            logger.info(f"{filing_path} is not html file, skipping...")
            return 0, []

        trimmed_html = _trim_html_content(filing_content)
        logger.debug(f"Trimmed HTML content size {len(trimmed_html)}")

        start_t = datetime.now()
        chunks = chunk_text(trimmed_html, method=method)
        elapsed_t = datetime.now() - start_t
        logger.debug(
            f"chunking with {len(filing_content)} of text with {method} took {elapsed_t.total_seconds()} seconds"  # noqa E501
        )

        if len(chunks) > 1:
            if table_name:
                logger.debug(f"Saving {len(chunks)} text chunks to {table_name}")
                save_chunks(
                    cik=filing.cik,
                    accession_number=filing.accession_number,
                    chunks=chunks,
                    table_name=table_name,
                    tags=tags,
                    create_table=True,
                )
            return len(chunks), chunks

    return 0, []


def chunk_text(
    content: str, chunk_size: int = DEFAULT_TEXT_CHUNK_SIZE, method: str = "spacy"
) -> list[str]:
    if method == "spacy":
        return _chunk_text_by_spacy(content, chunk_size)
    else:
        raise RuntimeError(f"Unknown method: {method}")


def _chunk_text_by_spacy(content: str, chunk_size: int) -> list[str]:
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

    # Remove empty chunks
    return [chunk for chunk in chunks if chunk.strip() and len(chunk.strip()) > 100]


def _trim_html_content(content: str) -> str:
    """
    remove the hidden div and convert the rest of html into text
    """
    if not content:
        return ""

    soup = BeautifulSoup(content, "html.parser")

    style_lambda = lambda value: value and "display:none" in value.replace(" ", "")  # noqa
    div_to_remove = soup.find("div", style=style_lambda)

    if div_to_remove:
        div_to_remove.decompose()  # type: ignore

    return _text2html(str(soup))


def _text2html(html_content: str):
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = True
    converter.ignore_emphasis = True
    converter.body_width = 0
    return converter.handle(html_content)
